# Copyright (C) 2026 Evandro Taquary and contributors
# Part of the pipeline accompanying "Enhancing Near-Real-Time Amazon Forest
# Monitoring Using GeoAI: A Case Study on Selective Logging".
# Licensed under the GNU Affero General Public License, version 3 or later.
# See the LICENSE file at the repository root, or <https://www.gnu.org/licenses/>.
"""TimeFirstAutoencoder — masked-pixel reconstruction over an HLS time series.

Architecture (high level)
-------------------------
We pivot away from next-frame forecasting to **MAE-style reconstruction**:
the dataset feeds L frames whose last slot IS the target with a synthetic
cloud mask applied; the model fills the masked pixels.

Pipeline:
    1. **Per-pixel tokenizer** (`PerPixelTokenizer`): 1×1 Conv2D shared
       across frames takes (B, L, 6, H, W) → (B, L, d_model, H, W).
       Channels: 0..3 reflectance, 4 aerosol level [0,1], 5 invalid-mask
       indicator (1 where the slot's reflectance is hidden / nodata,
       0 otherwise) — gives the model an explicit per-slot per-pixel
       valid/invalid signal.
    2. **Temporal embeddings**: a learned positional embedding over L slots
       PLUS a date-encoding MLP consuming `delta_days/365` are added to
       every per-pixel temporal token.
    3. **Temporal encoder**: `temporal_depth` pre-norm MHA blocks operating
       on (B·H·W, L, d_model) — i.e. one independent attention pass per
       pixel column. Bidirectional, no causal mask.
    4. **Time aggregation**: take the token at index `L-1` per pixel
       → (B, d_model, H, W). A learned **2D spatial positional embedding**
       (one tensor per pixel position) is added here so the spatial decoder
       has explicit per-pixel anchoring. Without it, the per-pixel temporal
       attention plus a vanilla Swin block produced spatially-uniform
       predictions because nothing distinguished similar-history pixels.
       This map is also the residual summed into every spatial block.
    5. **Spatial decoder**: `spatial_depth` Swin-style windowed-MHA blocks
       at the **original H×W** (no patch merging / no upsampling),
       alternating window-shift between (0,0) and (window/2, window/2).
    6. **Reconstruction head + identity bypass** (two modes, selected by
       `identity_bypass` ctor arg):

       - `"hard"` (default): content-aware `torch.where`:
             recon = where(fill_mask, head(feat), identity)
         `fill_mask` is the invalid-mask channel at slot L-1 of the input,
         driven directly by the dataset's synthetic mask M. Gradient flows
         into the head only at masked positions; unmasked positions return
         identity exactly. This is the strict bypass that avoids any drift
         on unmasked pixels.

       - `"per_band_per_pixel_alpha"` (ablation): a learnable
         `(out_channels, crop_size, crop_size)` α, sigmoid-bound, blends
         prediction with identity statically (no input dependence):
             recon = sigmoid(α) * head(feat) + (1 - sigmoid(α)) * identity
         Available for the C7 ablation. Note: with this gate the loss at
         masked pixels (where identity = 0) sees only sigmoid(α) * head;
         empirically α settles near 0.4-0.5 and visibly attenuates masked
         predictions. The "hard" mode avoids that.

NaN handling
------------
Reflectance NaNs (from nodata or the synthetic mask) are substituted by 0
**inside** `forward`, right after the input is split. The aerosol and
invalid-mask channels are always finite. Identity is computed AFTER
substitution, so unmasked pixels start at the original reflectance and
masked pixels start at 0 — the head's job is to fill the latter.

Loss
----
`masked_recon_loss(pred, target, mask, kind)` — L1 or L2 averaged over
`mask=True` pixels only (one band-mean across the 4 channels).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Tokenizer + embeddings
# ---------------------------------------------------------------------------


class PerPixelTokenizer(nn.Module):
    """1×1 Conv2D shared across frames: (B, L, C, H, W) → (B, L, D, H, W)."""

    def __init__(self, in_channels: int = 6, embed_dim: int = 192) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C, H, W)
        B, L, C, H, W = x.shape
        x = x.reshape(B * L, C, H, W)
        x = self.proj(x)
        D = x.shape[1]
        return x.reshape(B, L, D, H, W)


class DateEncoder(nn.Module):
    """2-layer MLP: delta_days (1) → d_model → d_model.

    Consumes `delta_days / 365` so a 1-year-old frame is ~1.0. The network
    learns the actual decay shape.
    """

    def __init__(self, embed_dim: int = 192, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or embed_dim
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, dates_days: torch.Tensor) -> torch.Tensor:
        # dates_days: (B, L)  — delta-days from target.
        x = (dates_days / 365.0).unsqueeze(-1)               # (B, L, 1)
        return self.net(x)                                   # (B, L, D)


# ---------------------------------------------------------------------------
# Temporal block: per-pixel MHA over time
# ---------------------------------------------------------------------------


class TemporalBlock(nn.Module):
    """Pre-norm: LN → MHA → res → LN → MLP → res. Bidirectional."""

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, L, D)  — N is the flattened "per-pixel" batch dim.
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Spatial block: Swin-style windowed attention, no downsampling
# ---------------------------------------------------------------------------


def _window_partition(x: torch.Tensor, window: int) -> torch.Tensor:
    """(B, H, W, D) → (B*nW, window*window, D)."""
    B, H, W, D = x.shape
    x = x.reshape(B, H // window, window, W // window, window, D)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.reshape(-1, window * window, D)


def _window_reverse(x: torch.Tensor, window: int, H: int, W: int) -> torch.Tensor:
    """(B*nW, window*window, D) → (B, H, W, D)."""
    B = x.shape[0] // ((H // window) * (W // window))
    x = x.reshape(B, H // window, W // window, window, window, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.reshape(B, H, W, -1)


class GlobalSpatialBlock(nn.Module):
    """Full-resolution spatial self-attention: every pixel attends to every
    pixel in the same crop, no window partition. Eliminates the 8×8 window
    seams the Swin block leaves. Quadratic in (H·W) tokens — on a 64×64
    crop that's 4096 tokens per attention sequence."""

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, H, W)
        B, D, H, W = x.shape
        x_hw = x.permute(0, 2, 3, 1).reshape(B, H * W, D)      # (B, H*W, D)
        residual = x_hw
        h = self.norm1(x_hw)
        a, _ = self.attn(h, h, h, need_weights=False)
        x_hw = residual + a
        x_hw = x_hw + self.mlp(self.norm2(x_hw))
        return x_hw.reshape(B, H, W, D).permute(0, 3, 1, 2).contiguous()


class WindowedSpatialBlock(nn.Module):
    """Swin block (no patch merging). Window MHA + MLP at full H×W.

    Window size and shift are configured per-block so the caller can
    alternate shifted / unshifted across the stack.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 6,
        window: int = 8,
        shift: int = 0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.window = window
        self.shift = shift
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, H, W)
        B, D, H, W = x.shape
        x_hw = x.permute(0, 2, 3, 1).contiguous()           # (B, H, W, D)
        residual = x_hw

        h = self.norm1(x_hw)
        if self.shift:
            h = torch.roll(h, shifts=(-self.shift, -self.shift), dims=(1, 2))

        wins = _window_partition(h, self.window)            # (B*nW, w*w, D)
        a, _ = self.attn(wins, wins, wins, need_weights=False)
        h = _window_reverse(a, self.window, H, W)           # (B, H, W, D)

        if self.shift:
            h = torch.roll(h, shifts=(self.shift, self.shift), dims=(1, 2))

        x_hw = residual + h
        x_hw = x_hw + self.mlp(self.norm2(x_hw))
        return x_hw.permute(0, 3, 1, 2).contiguous()        # (B, D, H, W)


# ---------------------------------------------------------------------------
# Reconstruction head
# ---------------------------------------------------------------------------


class ReconstructionHead(nn.Module):
    """1×1 Conv: d_model → out_channels."""

    def __init__(self, embed_dim: int = 192, out_channels: int = 4) -> None:
        super().__init__()
        self.proj = nn.Conv2d(embed_dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ---------------------------------------------------------------------------
# Time-first autoencoder
# ---------------------------------------------------------------------------


class TimeFirstAutoencoder(nn.Module):
    """Per-pixel temporal attention → windowed spatial attention.

    No patch merging anywhere — H×W is preserved end-to-end so the head
    has full-resolution information to fill the masked region.

    Parameters
    ----------
    in_channels
        Input channels per frame. Default 6 = 4 reflectance bands + 1
        normalised aerosol level + 1 invalid-mask indicator.
    out_channels
        Reconstruction channels. Default 4 (the reflectance bands).
    embed_dim, num_heads
        Token width and head count. Must satisfy `embed_dim % num_heads == 0`.
    temporal_depth, spatial_depth
        Number of TemporalBlocks and WindowedSpatialBlocks.
    seq_len
        Maximum sequence length L; sets the size of the learned temporal
        positional embedding.
    crop_size
        Side of the spatial input (H = W = crop_size). Sets the size of
        the learned 2D spatial positional embedding. The crop must be
        divisible by `window`.
    window
        Side length of the spatial attention window.
    mlp_ratio, dropout
        MLP expansion and dropout, applied uniformly.
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 4,
        embed_dim: int = 192,
        num_heads: int = 6,
        temporal_depth: int = 4,
        spatial_depth: int = 4,
        seq_len: int = 70,
        crop_size: int = 64,
        window: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        identity_bypass: str = "hard",
        spatial_attn: str = "global",
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if identity_bypass not in ("hard", "per_band_per_pixel_alpha"):
            raise ValueError(f"identity_bypass must be one of hard | per_band_per_pixel_alpha; got {identity_bypass!r}")
        if spatial_attn not in ("global", "window"):
            raise ValueError(f"spatial_attn must be 'global' or 'window'; got {spatial_attn!r}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.crop_size = crop_size
        self.window = window
        self.identity_bypass = identity_bypass
        self.spatial_attn = spatial_attn

        self.tokenizer = PerPixelTokenizer(in_channels, embed_dim)

        # Learned temporal positional embedding, (1, L, D).
        self.temporal_pe = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        nn.init.trunc_normal_(self.temporal_pe, std=0.02)

        # Learned 2D spatial positional embedding, (1, D, crop_size, crop_size).
        # Critical: per-pixel temporal attention alone produces spatially-uniform
        # outputs (similar histories → similar tokens), so the spatial decoder has
        # nothing to anchor on. This embedding gives every pixel a unique signature.
        self.spatial_pe = nn.Parameter(torch.zeros(1, embed_dim, crop_size, crop_size))
        nn.init.trunc_normal_(self.spatial_pe, std=0.02)

        self.date_encoder = DateEncoder(embed_dim)

        self.temporal_blocks = nn.ModuleList([
            TemporalBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(temporal_depth)
        ])
        self.temporal_norm = nn.LayerNorm(embed_dim)

        # Spatial decoder. "global" = every pixel attends to every pixel in
        # the crop (no 8×8 seams; quadratic in H·W). "window" = Swin-style
        # windowed MHA with alternating (0, w/2, 0, w/2, ...) shifts.
        if spatial_attn == "global":
            self.spatial_blocks = nn.ModuleList([
                GlobalSpatialBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(spatial_depth)
            ])
        else:
            self.spatial_blocks = nn.ModuleList([
                WindowedSpatialBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    window=window,
                    shift=(window // 2) if (i % 2 == 1) else 0,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for i in range(spatial_depth)
            ])
        self.spatial_norm_pre = nn.LayerNorm(embed_dim)
        self.head = ReconstructionHead(embed_dim, out_channels)

        # Per-band-per-pixel α gate is only allocated when identity_bypass
        # explicitly selects it (ablation C7). The "hard" default uses the
        # input's invalid-mask channel for a parameter-free, content-aware
        # torch.where bypass.
        if identity_bypass == "per_band_per_pixel_alpha":
            self.alpha = nn.Parameter(torch.zeros(out_channels, crop_size, crop_size))

    def forward(
        self,
        inputs: torch.Tensor,
        dates: torch.Tensor,
        window_phase: tuple[int, int] = (0, 0),
    ) -> torch.Tensor:
        """Reconstruct the (masked) target slot's reflectance.

        Parameters
        ----------
        inputs : (B, L, C_in, H, W) float32
            Channels 0..out_channels-1 are reflectance (NaN where invalid /
            masked); channel `out_channels` onward is auxiliary metadata
            (aerosol). NaN substitution to 0 happens here.
        dates : (B, L) float32
            Delta-days from the target frame. `dates[:, -1]` is 0.
        window_phase : (dy, dx) int tuple, default (0, 0)
            Inference-only: cyclically rolls the spatial feature map by
            (dy, dx) before the windowed-attention decoder, then rolls the
            result back — so the fixed Swin window grid lands at a different
            phase. Averaging outputs over several phases suppresses the 8×8
            window-boundary artefact. (0, 0) is an exact no-op (training path).

        Returns
        -------
        (B, out_channels, H, W) float32 — reconstructed reflectance.
        """
        B, L, C, H, W = inputs.shape
        if self.spatial_attn == "window" and (H % self.window or W % self.window):
            raise ValueError(
                f"H ({H}) and W ({W}) must be divisible by window ({self.window}) "
                f"when spatial_attn='window'"
            )

        # NaN → 0. Reflectance only — aerosol is always finite, but
        # nan_to_num is a no-op for finite tensors, so apply globally.
        x = torch.nan_to_num(inputs, nan=0.0)

        # Identity residual: target-slot reflectance (NaN→0). Used at the
        # end so unmasked pixels start at the original observation.
        identity = x[:, L - 1, : self.out_channels]                  # (B, C_out, H, W)

        # 1×1 Conv tokenization, shared over time.
        tokens = self.tokenizer(x)                                   # (B, L, D, H, W)
        D = tokens.shape[2]

        # Add temporal PE + date encoding (broadcast over pixels).
        date_emb = self.date_encoder(dates)                          # (B, L, D)
        temp_pe = self.temporal_pe[:, :L]                            # (1, L, D)
        bias = (temp_pe + date_emb).unsqueeze(-1).unsqueeze(-1)      # (B, L, D, 1, 1)
        tokens = tokens + bias

        # Reshape for temporal attention: (B, L, D, H, W) → (B*H*W, L, D).
        tokens = tokens.permute(0, 3, 4, 1, 2).contiguous()          # (B, H, W, L, D)
        tokens = tokens.reshape(B * H * W, L, D)
        for block in self.temporal_blocks:
            tokens = block(tokens)
        tokens = self.temporal_norm(tokens)

        # Take the target-slot token (index L-1).
        tgt = tokens[:, L - 1, :]                                    # (B*H*W, D)
        tgt = tgt.reshape(B, H, W, D).permute(0, 3, 1, 2).contiguous()  # (B, D, H, W)

        # Inject the learned 2D spatial PE so each pixel position has a unique
        # anchor for the spatial decoder. Sliced in case H, W < crop_size (padded crops).
        tgt = tgt + self.spatial_pe[:, :, :H, :W]

        # Inference-only window-phase shift: roll the feature map (carrying
        # its PE) so the fixed Swin window grid falls at a different phase,
        # rolled back after the decoder. Only meaningful for spatial_attn=
        # 'window'; under 'global' attention there are no windows to alias
        # against, so the roll is a pointless no-op (skipped).
        dy, dx = window_phase
        if (dy or dx) and self.spatial_attn == "window":
            tgt = torch.roll(tgt, shifts=(dy, dx), dims=(2, 3))

        # Spatial decoder. Skip from `tgt` summed into every block input.
        feat = tgt
        feat = self.spatial_norm_pre(
            feat.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2).contiguous()
        for block in self.spatial_blocks:
            feat = block(feat + tgt)

        if dy or dx:
            feat = torch.roll(feat, shifts=(-dy, -dx), dims=(2, 3))

        pred = self.head(feat)                                       # (B, C, H, W)
        if self.identity_bypass == "per_band_per_pixel_alpha":
            # Static learnable blend, ablation only (see class docstring).
            a = self.alpha[:, :H, :W].sigmoid().unsqueeze(0)         # (1, C, H, W)
            return a * pred + (1.0 - a) * identity
        # Default: content-aware hard gate driven by the invalid-mask channel
        # at slot L-1 — exactly where the synthetic mask was applied at
        # training time. The dataset always puts invalid-mask as the LAST
        # channel (index in_channels - 1) so this works whether the optional
        # aero channel is included or not. Unmasked positions return
        # `identity` (the original reflectance), so the head has gradient
        # only where we asked it to predict.
        fill_mask = (x[:, L - 1, self.in_channels - 1] > 0.5).unsqueeze(1)  # (B, 1, H, W)
        return torch.where(fill_mask, pred, identity)

    def forward_features(
        self,
        inputs: torch.Tensor,
        dates: torch.Tensor,
        window_phase: tuple[int, int] = (0, 0),
    ) -> torch.Tensor:
        """Per-pixel features for downstream heads (no reconstruction).

        Runs the encoder + spatial decoder but stops BEFORE the reconstruction
        head, returning the concatenation along channels of two views of the
        target-slot per-pixel embedding:
          - tgt_pure: pure-temporal — token at slot L-1 after `temporal_blocks`
            + `temporal_norm`, no spatial PE, no spatial mixing.  (B, D, H, W)
          - feat: spatial+temporal — after `spatial_blocks` with global
            attention mixing every pixel in the crop.  (B, D, H, W)

        Concatenated along channels so a downstream head sees both views.

        Returns
        -------
        (B, 2D, H, W) float — concat([tgt_pure, feat], dim=1). D = embed_dim.
        """
        B, L, C, H, W = inputs.shape
        if self.spatial_attn == "window" and (H % self.window or W % self.window):
            raise ValueError(
                f"H ({H}) and W ({W}) must be divisible by window ({self.window}) "
                f"when spatial_attn='window'"
            )

        x = torch.nan_to_num(inputs, nan=0.0)

        tokens = self.tokenizer(x)
        D = tokens.shape[2]
        date_emb = self.date_encoder(dates)
        temp_pe = self.temporal_pe[:, :L]
        bias = (temp_pe + date_emb).unsqueeze(-1).unsqueeze(-1)
        tokens = tokens + bias

        tokens = tokens.permute(0, 3, 4, 1, 2).contiguous()
        tokens = tokens.reshape(B * H * W, L, D)
        for block in self.temporal_blocks:
            tokens = block(tokens)
        tokens = self.temporal_norm(tokens)

        tgt_pure = tokens[:, L - 1, :].reshape(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        # Same spatial decoder path as forward() — operates on tgt + spatial_pe.
        tgt_pe = tgt_pure + self.spatial_pe[:, :, :H, :W]

        dy, dx = window_phase
        if (dy or dx) and self.spatial_attn == "window":
            tgt_pe = torch.roll(tgt_pe, shifts=(dy, dx), dims=(2, 3))

        feat = self.spatial_norm_pre(
            tgt_pe.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2).contiguous()
        for block in self.spatial_blocks:
            feat = block(feat + tgt_pe)

        if (dy or dx) and self.spatial_attn == "window":
            feat = torch.roll(feat, shifts=(-dy, -dx), dims=(2, 3))

        return torch.cat([tgt_pure, feat], dim=1)


# ---------------------------------------------------------------------------
# Loss helper
# ---------------------------------------------------------------------------


def masked_recon_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    kind: str = "l1",
) -> torch.Tensor:
    """Reconstruction loss over `mask=True` pixels only.

    pred, target : (B, C, H, W) float
    mask         : (B, H, W) bool  — broadcasts over channels
    kind         : "l1" | "l2"
    """
    diff = pred - target
    if kind == "l1":
        per_pix = diff.abs()
    elif kind == "l2":
        per_pix = diff.pow(2)
    else:
        raise ValueError(f"unknown loss kind: {kind!r}")

    m = mask.unsqueeze(1).to(per_pix.dtype)                          # (B, 1, H, W)
    num = (per_pix * m).sum()
    denom = (m.sum() * pred.shape[1]).clamp(min=1.0)
    return num / denom


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, H, W, C_in, C_out = 2, 8, 64, 64, 8, 6
    model = TimeFirstAutoencoder(
        in_channels=C_in, out_channels=C_out,
        embed_dim=96, num_heads=3,
        temporal_depth=3, spatial_depth=2,
        seq_len=L, crop_size=H, window=8, mlp_ratio=4.0, dropout=0.15,
    )
    model.eval()

    inputs = torch.randn(B, L, C_in, H, W)
    # Drop ~30% to NaN on reflectance to mimic the dataset; populate the
    # invalid-mask channel (index C_out+1) accordingly so it tracks the NaN
    # footprint (used elsewhere in train / diagnostics).
    nan_mask = torch.rand(B, L, C_out, H, W) < 0.30
    inputs[..., :C_out, :, :][nan_mask] = float("nan")
    inputs[..., C_out + 1, :, :] = nan_mask.any(dim=2).float()
    dates = torch.linspace(60, 0, L).expand(B, L).contiguous()
    target = torch.rand(B, C_out, H, W) * 0.3
    mask = torch.rand(B, H, W) < 0.5

    with torch.no_grad():
        pred = model(inputs, dates)
    n_params = sum(p.numel() for p in model.parameters())
    n_alpha = model.alpha.numel() if hasattr(model, "alpha") else 0
    print(f"params: {n_params:,} (alpha gate: {n_alpha:,}; bypass mode: {model.identity_bypass})")
    print(f"pred  : {tuple(pred.shape)}  dtype={pred.dtype}  any_nan={torch.isnan(pred).any().item()}")

    # With α initialised to zeros, sigmoid(α) = 0.5 → recon = 0.5 * pred + 0.5 * identity.
    # Validate roughly: take the average over masked pixels and confirm finite.
    loss_l1 = masked_recon_loss(pred, target, mask, kind="l1")
    loss_l2 = masked_recon_loss(pred, target, mask, kind="l2")
    print(f"loss(mask, l1) = {loss_l1.item():.4f}")
    print(f"loss(mask, l2) = {loss_l2.item():.6f}")
