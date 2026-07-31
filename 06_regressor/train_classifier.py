# Copyright (C) 2026 Evandro Taquary and contributors
# Part of the pipeline accompanying "Enhancing Near-Real-Time Amazon Forest
# Monitoring Using GeoAI: A Case Study on Selective Logging".
# Licensed under the GNU Affero General Public License, version 3 or later.
# See the LICENSE file at the repository root, or <https://www.gnu.org/licenses/>.
"""train_classifier.py — soft-label REGRESSION head over the frozen/partial-unfreeze
foundation trunk. Targets are the per-ROI soft labels v in [0,1] (decay from seeds);
loss = MSE(sigmoid(logit), v). LOOCV by held-out positive ROI (--loo-roi); cross-ROI
negatives (fold_roi="__cross__") are training-only. Per-variant or combined input.

Usage (one fold):
    python train_classifier.py --variant raw --loo-roi 11984_curr \\
        --branch raw-loo11984_curr --unfreeze-spatial --bf16 \\
        --lr 1e-5 --weight-decay 0.1 --head-hidden 256
"""
from __future__ import annotations
import argparse, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_pretraining"))
from classifier_head import ClassificationHead
from dataset_classifier import HLSClassifierDataset, collate, load_annotations
from model import TimeFirstAutoencoder

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

VARIANTS = {"raw": ("raw",), "recon": ("recon",), "masked": ("masked",),
            "composite": ("composite",), "all": ("raw", "recon", "masked", "composite")}

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--variant", choices=list(VARIANTS), required=True)
    p.add_argument("--loo-roi", type=str, required=True, help="Held-out positive ROI (val).")
    p.add_argument("--branch", type=str, required=True)
    p.add_argument("--zarr-dir", type=Path, default=DATA_DIR / "labelled-dataset")
    p.add_argument("--annotations", type=Path, default=DATA_DIR / "annotations.json")
    p.add_argument("--view-dates", type=Path, default=DATA_DIR / "view_dates.json")
    p.add_argument("--foundation-ckpt", type=Path, default=DATA_DIR / "checkpoints" / "foundation" / "best.pt")
    p.add_argument("--runs-root", type=Path, default=DATA_DIR / "runs")
    p.add_argument("--freeze-trunk", action="store_true")
    p.add_argument("--unfreeze-spatial", action="store_true")
    p.add_argument("--max-epochs", type=int, default=120)
    p.add_argument("--patience-epochs", type=int, default=35)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--head-hidden", type=int, default=256)
    p.add_argument("--head-dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--resume-from", type=Path, default=None,
                   help="Resume trunk+head+optimizer+epoch from this checkpoint (e.g. .../last.pt). "
                        "Patience is reset to --patience-epochs so a previously early-stopped fold can "
                        "keep improving. Use the SAME unfreeze/arch config as the original run.")
    p.add_argument("--exclude-rois", type=str, default="",
                   help="Comma-separated positive ROIs to exclude from the experiment ENTIRELY "
                        "(their positives, within-ROI negatives, and cross-neg slices are dropped "
                        "from every fold's training and from the ROI pool).")
    return p.parse_args()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def _trunk_arch_kwargs(sa):
    keys = ("embed_dim", "num_heads", "temporal_depth", "spatial_depth", "in_channels",
            "out_channels", "crop_size", "seq_len", "dropout", "mlp_ratio", "window",
            "identity_bypass", "spatial_attn")
    src = dict(sa)
    if "seq_len" not in src and "lookback" in src: src["seq_len"] = src["lookback"]
    return {k: src[k] for k in keys if k in src}


def load_trunk(ckpt, device, freeze_all, unfreeze_spatial):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sa = ck.get("args", {})
    trunk = TimeFirstAutoencoder(**_trunk_arch_kwargs(sa))
    state = ck["model"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    trunk.load_state_dict(state, strict=True)
    if freeze_all:
        for p in trunk.parameters(): p.requires_grad_(False)
        trunk.eval()
    elif unfreeze_spatial:
        for p in trunk.parameters(): p.requires_grad_(False)
        for m in (trunk.spatial_blocks, trunk.spatial_norm_pre):
            for p in m.parameters(): p.requires_grad_(True)
        trunk.spatial_pe.requires_grad_(True)
    trunk.to(device)
    return trunk, sa


def atomic_save(state, path):
    tmp = path.with_suffix(".pt.tmp"); torch.save(state, tmp)
    with tmp.open("rb") as fh: os.fsync(fh.fileno())
    tmp.replace(path)


def reg_metrics(pred, tgt):
    pred = np.asarray(pred); tgt = np.asarray(tgt)
    err = pred - tgt
    pos = tgt > 0; neg = ~pos
    return {"mse": float((err ** 2).mean()), "mae": float(np.abs(err).mean()),
            "mse_pos": float((err[pos] ** 2).mean()) if pos.any() else float("nan"),
            "mse_neg": float((err[neg] ** 2).mean()) if neg.any() else float("nan"),
            "n_pos": int(pos.sum()), "n_neg": int(neg.sum())}


def main():
    args = parse_args(); set_seed(args.seed)
    import json
    from datetime import date
    run_dir = args.runs_root / args.branch; ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(
        {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, sort_keys=True),
        encoding="utf-8")
    val_log = run_dir / "val_history.csv"
    if not val_log.exists():
        val_log.write_text("epoch,train_loss,val_mse,val_mae,val_mse_pos,val_mse_neg,n_pos,n_neg\n",
                            encoding="utf-8")

    ps, amb, pool, refs = load_annotations(args.annotations)
    vds = {a: date.fromisoformat(s) for a, s in json.loads(args.view_dates.read_text(encoding="utf-8")).items()}
    excl = set(a for a in args.exclude_rois.split(",") if a)
    ds = HLSClassifierDataset(args.zarr_dir, ps, amb, refs, vds, oversample_pool=pool,
                                variants=VARIANTS[args.variant], exclude_rois=excl, seed=args.seed)
    if args.loo_roi not in ds.activity_rois:
        raise SystemExit(f"--loo-roi {args.loo_roi} not among positive ROIs {ds.activity_rois}")
    train_idx, val_idx = ds.fold_indices(args.loo_roi)
    print(f"[data] variant={args.variant} samples={len(ds)} (pos {ds.n_pos_base}, neg {ds.n_neg_base})", flush=True)
    print(f"[loo] held-out={args.loo_roi}  train={len(train_idx)}  val={len(val_idx)}", flush=True)
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
                              prefetch_factor=4 if args.num_workers > 0 else None,
                              collate_fn=collate, drop_last=False)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=args.batch_size, shuffle=False,
                            num_workers=max(2, args.num_workers // 3), persistent_workers=args.num_workers > 0,
                            collate_fn=collate, drop_last=False)

    trunk, ta = load_trunk(args.foundation_ckpt, args.device, args.freeze_trunk, args.unfreeze_spatial)
    embed_dim = int(ta.get("embed_dim", 64))
    head = ClassificationHead(embed_dim=embed_dim, hidden=args.head_hidden, dropout=args.head_dropout).to(args.device)
    n_train = (sum(p.numel() for p in trunk.parameters() if p.requires_grad)
               + sum(p.numel() for p in head.parameters()))
    print(f"[model] embed={embed_dim} trainable={n_train:,}", flush=True)
    params = [p for p in trunk.parameters() if p.requires_grad] + list(head.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    def forward(batch):
        x = batch["inputs"].to(args.device, non_blocking=True)
        d = batch["dates"].to(args.device, non_blocking=True)
        if args.bf16 and args.device == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feats = trunk.forward_features(x, d)
            logits = head(feats.float())
        else:
            logits = head(trunk.forward_features(x, d))
        return torch.sigmoid(logits)

    start_epoch, best_mse, best_epoch, patience = 0, float("inf"), -1, args.patience_epochs
    if args.resume_from:
        rck = torch.load(args.resume_from, map_location=args.device, weights_only=False)
        head.load_state_dict(rck["head"])
        if not args.freeze_trunk and "trunk" in rck:
            trunk.load_state_dict(rck["trunk"])
        if "optimizer" in rck:
            optim.load_state_dict(rck["optimizer"])
        rs = rck.get("rng")
        if rs:
            random.setstate(rs["python"]); np.random.set_state(rs["numpy"])
            torch.set_rng_state(rs["torch"].cpu())   # RNG state must be a CPU ByteTensor
            if rs.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([t.cpu() for t in rs["cuda"]])
        start_epoch = int(rck.get("epoch", -1)) + 1
        best_mse, best_epoch = float(rck.get("best_mse", float("inf"))), int(rck.get("best_epoch", -1))
        patience = args.patience_epochs   # fresh budget so a previously early-stopped fold can keep improving
        print(f"[resume] {args.resume_from}: start ep{start_epoch}, best_mse={best_mse:.5f} @ ep{best_epoch}", flush=True)
    for epoch in range(start_epoch, args.max_epochs):
        trunk.train(); head.train(); tot, n = 0.0, 0; t0 = time.time()
        for batch in train_loader:
            tgt = batch["target"].to(args.device, non_blocking=True)
            pred = forward(batch)
            loss = F.mse_loss(pred, tgt)
            optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
            tot += float(loss) * tgt.size(0); n += tgt.size(0)
        train_loss = tot / max(n, 1); train_dt = time.time() - t0

        trunk.eval(); head.eval(); preds, tgts = [], []; t0 = time.time()
        with torch.no_grad():
            for batch in val_loader:
                preds.append(forward(batch).cpu().numpy()); tgts.append(batch["target"].numpy())
        preds = np.concatenate(preds); tgts = np.concatenate(tgts)
        m = reg_metrics(preds, tgts); val_dt = time.time() - t0
        print(f"[ep {epoch:03d}] train={train_loss:.5f} val_mse={m['mse']:.5f} mae={m['mae']:.5f} "
              f"mse_pos={m['mse_pos']:.5f} mse_neg={m['mse_neg']:.5f} t={train_dt:.1f}+{val_dt:.1f}s", flush=True)
        with val_log.open("a", encoding="utf-8") as fh:
            fh.write(f"{epoch},{train_loss:.6f},{m['mse']:.6f},{m['mae']:.6f},{m['mse_pos']:.6f},"
                     f"{m['mse_neg']:.6f},{m['n_pos']},{m['n_neg']}\n")

        improved = m["mse"] < best_mse
        if improved:
            best_mse, best_epoch, patience = m["mse"], epoch, args.patience_epochs
        else:
            patience -= 1
        rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
               "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        state = {"head": head.state_dict(), "optimizer": optim.state_dict(), "rng": rng,
                 "args": vars(args), "epoch": epoch, "best_mse": best_mse, "best_epoch": best_epoch,
                 "patience": patience, "embed_dim": embed_dim, "trunk_arch": _trunk_arch_kwargs(ta),
                 "foundation_ckpt": str(args.foundation_ckpt), "variant": args.variant,
                 "loo_roi": args.loo_roi, "seed": args.seed, "val_mse": m["mse"],
                 "val_preds": preds.tolist(), "val_tgts": tgts.tolist()}
        if not args.freeze_trunk:
            state["trunk"] = trunk.state_dict()
        atomic_save(state, ckpt_dir / "last.pt")
        if improved:
            atomic_save(state, ckpt_dir / "best.pt")
            print(f"  -> new best val MSE = {best_mse:.5f}", flush=True)
        if patience <= 0:
            print(f"  -> early stop at ep {epoch}", flush=True); break
    print(f"\n[done] {args.variant} loo {args.loo_roi}: best val MSE = {best_mse:.5f} @ ep{best_epoch}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
