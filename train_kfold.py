"""
K-fold cross-validation wrapper for the equivariant protein detector.

The default training split holds out a single run (TS_99_9), which gives a
high-variance F-beta estimate.  This script runs leave-one-out (or arbitrary
k-fold) CV across all discovered TS_* runs, training one model per fold
and aggregating the final / best F-beta scores into a JSON report.

It launches train.py once per fold (as a subprocess) so DDP and CUDA state
are reset between folds, and so any choice of --model-variant /
--learnable-z-sigma works without modification.

Per-fold val run is communicated to train.py via the environment variable
CZII_VAL_RUN — train.py reads it and overrides cfg.data.val_run_names.

Usage
-----
Single-GPU leave-one-out:
    python train_kfold.py --model-variant so3 --epochs 100

H100x2:
    python train_kfold.py --model-variant so3 \\
        --launcher "torchrun --nproc_per_node=2" \\
        --epochs 100

K-fold (groups of 2 runs per fold):
    python train_kfold.py --model-variant so3 --k-folds 3 --epochs 100

The per-fold checkpoints land in:
    {checkpoint_dir}/{variant}/fold_{idx:02d}_val_{run_name}/

The aggregated report is written to:
    kfold_results_{variant}.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import torch

from config import Config
from dataset import discover_run_names


def make_folds(runs: List[str], k_folds: int) -> List[List[str]]:
    """Split runs into k contiguous-ish val groups.

    k_folds == len(runs) is leave-one-out.  Larger groups when k_folds is
    smaller than the number of runs.
    """
    if k_folds <= 0 or k_folds > len(runs):
        k_folds = len(runs)
    # Even distribution; the last fold absorbs any remainder.
    base = len(runs) // k_folds
    folds = []
    start = 0
    for i in range(k_folds):
        end = start + base if i < k_folds - 1 else len(runs)
        folds.append(runs[start:end])
        start = end
    return folds


def run_one_fold(
    train_script: Path,
    launcher: str,
    extra_args: List[str],
    val_runs: List[str],
    checkpoint_dir: Path,
    log_dir: Path,
    env_extra: dict,
) -> int:
    """Launch train.py for one fold.  Returns subprocess return code."""
    env = os.environ.copy()
    # train.py reads CZII_VAL_RUN to override cfg.data.val_run_names
    env["CZII_VAL_RUN"] = ",".join(val_runs)
    env["CZII_CKPT_DIR"] = str(checkpoint_dir)
    env["CZII_LOG_DIR"] = str(log_dir)
    env.update(env_extra)

    cmd = launcher.split() + [str(train_script)] + extra_args
    print(f"\n[fold] val={val_runs}  cmd={' '.join(cmd)}")
    completed = subprocess.run(cmd, env=env, check=False)
    return completed.returncode


def read_best_fbeta(ckpt_path: Path) -> float:
    """Return val F-beta stored in a best.pt checkpoint (or NaN if missing)."""
    if not ckpt_path.exists():
        return float("nan")
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        metrics = ckpt.get("metrics", {})
        # Prefer the full-volume metric if present, else patch-level.
        return float(
            metrics.get("full_val_fbeta",
            metrics.get("val_fbeta", float("nan")))
        )
    except Exception as exc:
        print(f"  ! Could not read {ckpt_path}: {exc}")
        return float("nan")


def main():
    parser = argparse.ArgumentParser(description="K-fold CV wrapper for train.py")
    parser.add_argument("--dataset", default=None,
                        help="Dataset root (default: from Config.data.dataset_dir)")
    parser.add_argument("--model-variant", default="so3",
                        choices=["so3", "so3_aniso", "so2", "so2_iso"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--learnable-z-sigma", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--k-folds", type=int, default=0,
                        help="Number of folds (0 = leave-one-out across all runs)")
    parser.add_argument("--launcher", default="python",
                        help="Launcher prefix, e.g. 'python' or 'torchrun --nproc_per_node=2'")
    parser.add_argument("--output", default=None,
                        help="Path to write aggregated results JSON")
    parser.add_argument("--skip-existing", action="store_true",
                        help="If a fold's best.pt already exists, do not retrain")
    args = parser.parse_args()

    cfg = Config()
    dataset_dir = args.dataset or cfg.data.dataset_dir
    all_runs = discover_run_names(dataset_dir)
    if not all_runs:
        sys.exit(f"No TS_* runs found in {dataset_dir}")

    folds = make_folds(all_runs, args.k_folds or len(all_runs))
    variant = "baseline" if args.baseline else args.model_variant

    train_script = Path(__file__).parent / "train.py"
    ckpt_root = Path(cfg.train.checkpoint_dir) / variant
    log_root  = Path(cfg.train.log_dir) / variant
    out_path  = Path(args.output or f"kfold_results_{variant}.json")

    # Build the per-fold argument list once.
    extra_args: List[str] = []
    if args.epochs is not None:
        extra_args += ["--epochs", str(args.epochs)]
    if args.batch_size is not None:
        extra_args += ["--batch-size", str(args.batch_size)]
    if args.lr is not None:
        extra_args += ["--lr", str(args.lr)]
    if args.baseline:
        extra_args += ["--baseline"]
    else:
        extra_args += ["--model-variant", args.model_variant]
    if args.learnable_z_sigma:
        extra_args += ["--learnable-z-sigma"]

    results = []
    t_start = time.time()
    for fold_idx, val_runs in enumerate(folds):
        fold_tag = f"fold_{fold_idx:02d}_val_{'_'.join(val_runs)}"
        ckpt_dir = ckpt_root / fold_tag
        log_dir  = log_root  / fold_tag
        best_pt  = ckpt_dir / "best.pt"

        if args.skip_existing and best_pt.exists():
            print(f"\n[fold {fold_idx}] best.pt exists at {best_pt} — skipping")
        else:
            t0 = time.time()
            rc = run_one_fold(
                train_script=train_script,
                launcher=args.launcher,
                extra_args=extra_args,
                val_runs=val_runs,
                checkpoint_dir=ckpt_dir,
                log_dir=log_dir,
                env_extra={},
            )
            if rc != 0:
                print(f"[fold {fold_idx}] train.py exited with code {rc}")
            print(f"[fold {fold_idx}] elapsed {time.time() - t0:.1f}s")

        fb = read_best_fbeta(best_pt)
        results.append({
            "fold":            fold_idx,
            "val_runs":        val_runs,
            "checkpoint":      str(best_pt),
            "best_val_fbeta":  fb,
        })

    # Aggregate
    fbs = [r["best_val_fbeta"] for r in results
           if r["best_val_fbeta"] == r["best_val_fbeta"]]  # filter NaN
    mean_fb = sum(fbs) / len(fbs) if fbs else float("nan")
    summary = {
        "variant":         variant,
        "n_folds":         len(folds),
        "mean_val_fbeta":  mean_fb,
        "std_val_fbeta":   (
            (sum((x - mean_fb) ** 2 for x in fbs) / max(len(fbs) - 1, 1)) ** 0.5
            if len(fbs) > 1 else 0.0
        ),
        "elapsed_seconds": time.time() - t_start,
        "folds":           results,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 60)
    print(f"k-fold CV complete: {len(fbs)}/{len(folds)} folds scored")
    print(f"  mean F-beta = {mean_fb:.4f}  (std {summary['std_val_fbeta']:.4f})")
    print(f"  per-fold:")
    for r in results:
        print(f"    fold {r['fold']:2d}  val={r['val_runs']}  F-beta={r['best_val_fbeta']:.4f}")
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
