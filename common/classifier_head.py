# Copyright (C) 2026 Evandro Taquary and contributors
# Part of the pipeline accompanying "Enhancing Near-Real-Time Amazon Forest
# Monitoring Using GeoAI: A Case Study on Selective Logging".
# Licensed under the GNU Affero General Public License, version 3 or later.
# See the LICENSE file at the repository root, or <https://www.gnu.org/licenses/>.
"""Classification head for the foundation-model-based binary classifier.

Consumes the concatenated per-pixel features produced by
`TimeFirstAutoencoder.forward_features()` — shape `(B, 2*embed_dim, H, W)` —
and emits one logit per crop (binary classification).

Architecture (small on purpose; the trunk is doing the heavy lifting):
    AdaptiveAvgPool2d(1) → flatten → Linear(2D, 128) → GELU → Dropout(p) → Linear(128, 1)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """MLP head over globally-pooled trunk features.

    Parameters
    ----------
    embed_dim : int
        D of the foundation model. The head expects feature maps of channel
        size `2 * embed_dim` (since `forward_features` concats `tgt_pure` and
        `feat`).
    hidden : int, default 128
        Width of the hidden MLP layer.
    dropout : float, default 0.1
        Dropout probability between the hidden and output layers.
    """

    def __init__(self, embed_dim: int, hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        in_dim = 2 * embed_dim
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, 2D, H, W) → (B,) logit."""
        pooled = self.pool(features).flatten(1)   # (B, 2D)
        return self.mlp(pooled).squeeze(-1)       # (B,)
