"""Learning-curve study: how does AP50 scale with training-set size?

Runs `src/train.py` repeatedly at multiple subsample levels of the training
set (geometric grid over sequence count) against a fixed held-out val set,
then plots AP50 vs #sequences and AP50 vs #images side-by-side.

Usage — wrapper flags first (--train-dir, --grid, etc.); any remaining
arguments are forwarded as Hydra overrides to train.py exactly as you'd
write them on a `python src/train.py ...` command line:

    python src/learning_curve.py --train-dir omni_2.17_train \\
        model=faster_rcnn \\
        training.num_epochs=50 \\
        'classes=[human_annotated_positive_fish_blob,triangle]' \\
        'model.transform.input_size=[480,640]' \\
        'dataset.normalization.image_mean=[0.047306,0.042015,0.444843]' \\
        'dataset.normalization.image_std=[0.140571,0.134107,0.159125]'

Output layout (under outputs/learning_curve_<timestamp>/):
    val_holdout/         — held-out val set (jsonl + symlinked images)
    runs/n_seq=K/        — full Hydra run dir per training job
    manifest.json        — sequence-id assignments, grid, seeds
    results.json         — [{n_sequences, n_images, AP50, AP, AP75, run_dir}, ...]
    learning_curve.png   — the plot
"""
import argparse
import json
import logging
import os
import random
import shlex
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [learning_curve] %(levelname)s %(message)s',
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-dir", required=True,
                   help="Path to full training dataset (must contain dataset.jsonl + data/)")
    p.add_argument("--output-root", default="outputs",
                   help="Where to put learning_curve_<ts>/ (default: outputs)")
    p.add_argument("--val-fraction", type=float, default=0.2,
                   help="Fraction of sequences held out for val (default: 0.2)")
    p.add_argument("--grid", nargs="+", type=int, default=None,
                   help="Explicit list of n_sequences (overrides geometric grid)")
    p.add_argument("--grid-points", type=int, default=6,
                   help="Number of geometric grid points if --grid not given (default: 6)")
    p.add_argument("--min-sequences", type=int, default=2,
                   help="Smallest grid point if --grid not given (default: 2)")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for val/pool/grid-subset selection (default: 42)")
    p.add_argument("--python", default=sys.executable,
                   help="Python interpreter to invoke for train.py (default: current)")
    # Any remaining args are Hydra-style overrides (key=value) forwarded to train.py.
    # Use parse_known_args so the user can pass them inline with the same syntax
    # they'd use when calling train.py directly.
    args, train_overrides = p.parse_known_args()
    args.train_overrides = train_overrides
    return args


def extract_sequence_id(record: dict) -> str | None:
    """Pull the sequence_id out of a JSONL record (mirrors ViamDataset)."""
    for ann in record.get('classification_annotations', []):
        label = ann.get('annotation_label', '')
        if label.startswith('sequence_'):
            return label.split('--')[0]
    return None


def discover_sequences(jsonl_path: Path) -> dict[str, list[int]]:
    """Group line numbers by sequence_id. Skips records without a sequence_id."""
    seqs: dict[str, list[int]] = defaultdict(list)
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            seq_id = extract_sequence_id(data)
            if seq_id is None:
                continue
            seqs[seq_id].append(i)
    return dict(seqs)


def build_val_holdout(
    src_jsonl: Path, src_data_dir: Path, val_seq_ids: set[str], dest: Path
) -> int:
    """Materialize a held-out val dir at `dest`: filtered jsonl + symlinked images.

    Returns number of images materialized.
    """
    dest.mkdir(parents=True, exist_ok=True)
    dest_data = dest / "data"
    dest_data.mkdir(exist_ok=True)
    dest_jsonl = dest / "dataset.jsonl"

    n = 0
    with open(src_jsonl) as fin, open(dest_jsonl, "w") as fout:
        for line in fin:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if extract_sequence_id(data) not in val_seq_ids:
                continue

            fout.write(line if line.endswith("\n") else line + "\n")
            n += 1

            # Resolve source image path (mirrors ViamDataset.__getitem__ logic)
            image_path = data.get('image_path')
            if not image_path:
                continue
            if os.path.isabs(image_path):
                src_img = Path(image_path)
            elif image_path.startswith(src_data_dir.name + '/'):
                src_img = src_data_dir.parent / image_path
            else:
                src_img = src_data_dir / os.path.basename(image_path)

            link = dest_data / os.path.basename(image_path)
            if not link.exists():
                try:
                    link.symlink_to(src_img.resolve())
                except FileExistsError:
                    pass

    return n


def make_geometric_grid(max_seq: int, points: int, min_seq: int) -> list[int]:
    """Geometric-spaced integer grid from min_seq to max_seq, inclusive, deduped."""
    min_seq = max(1, min(min_seq, max_seq))
    if max_seq <= min_seq:
        return [max_seq]
    raw = np.geomspace(min_seq, max_seq, num=points)
    grid = sorted({int(round(x)) for x in raw})
    if grid[-1] != max_seq:
        grid.append(max_seq)
    return grid


def hydra_list(items: list[str]) -> str:
    """Format a list of strings as a Hydra/OmegaConf list literal."""
    # Sequence IDs are alphanumeric + underscore + dash. No quoting needed for those,
    # but wrap defensively in case future IDs contain commas or spaces.
    return "[" + ",".join(items) + "]"


def run_training(
    python_bin: str,
    repo_root: Path,
    train_dir: Path,
    val_dir: Path,
    train_seq_ids: list[str],
    run_dir: Path,
    extra_overrides: list[str],
) -> bool:
    """Invoke `python src/train.py ...` for one grid point. Returns True on success."""
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_bin,
        "src/train.py",
        f"hydra.run.dir={run_dir}",
        f"dataset.data.train_dir={train_dir}",
        f"dataset.data.val_dir={val_dir}",
        f"dataset.data.train_sequence_ids={hydra_list(train_seq_ids)}",
    ]
    cmd.extend(extra_overrides)

    log.info("Launching training run:")
    log.info("  cwd: %s", repo_root)
    log.info("  cmd: %s", " ".join(shlex.quote(c) for c in cmd))

    result = subprocess.run(cmd, cwd=repo_root)
    return result.returncode == 0


def plot_curve(results: list[dict], out_path: Path) -> None:
    """Two subplots: AP50 vs #sequences, AP50 vs #images. Log x."""
    if not results:
        log.warning("No results to plot.")
        return

    results_sorted = sorted(results, key=lambda r: r["n_sequences"])
    n_seq = [r["n_sequences"] for r in results_sorted]
    n_img = [r["n_images"] for r in results_sorted]
    ap50 = [r["AP50"] for r in results_sorted]
    ap = [r["AP"] for r in results_sorted]
    ap75 = [r["AP75"] for r in results_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, x, xlabel, title in [
        (axes[0], n_seq, "# Training Sequences", "Learning curve — by sequence"),
        (axes[1], n_img, "# Training Images", "Learning curve — by image"),
    ]:
        ax.plot(x, ap50, 'o-', linewidth=2.5, color='black', label='AP50', zorder=3)
        ax.plot(x, ap, 's--', linewidth=1.2, color='tab:blue', alpha=0.7, label='AP (50:95)')
        ax.plot(x, ap75, '^--', linewidth=1.2, color='tab:orange', alpha=0.7, label='AP75')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Validation metric", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.set_xscale('log')
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(loc='lower right')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("Saved plot: %s", out_path)


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    repo_root = Path(__file__).resolve().parent.parent
    train_dir = Path(args.train_dir).resolve()
    train_jsonl = train_dir / "dataset.jsonl"
    train_data_dir = train_dir / "data"

    if not train_jsonl.exists():
        sys.exit(f"dataset.jsonl not found in {train_dir}")
    if not train_data_dir.exists():
        sys.exit(f"data/ directory not found in {train_dir}")

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_root = (Path(args.output_root) / f"learning_curve_{ts}").resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    log.info("Output dir: %s", out_root)

    # 1. Discover sequences (skip records without sequence_id — excluded entirely
    #    from this study, per design).
    seqs = discover_sequences(train_jsonl)
    if not seqs:
        sys.exit("No sequence_ids found in training set — cannot build a learning curve.")
    all_seq_ids = sorted(seqs.keys())
    log.info("Discovered %d sequences (%d images with sequence_id)",
             len(all_seq_ids), sum(len(v) for v in seqs.values()))

    # 2. Pick held-out val sequences
    shuffled = list(all_seq_ids)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * args.val_fraction)))
    val_seq_ids = set(shuffled[:n_val])
    pool_seq_ids = shuffled[n_val:]  # keep shuffled order — used for nested subsetting
    if not pool_seq_ids:
        sys.exit("After val holdout, no sequences left for training pool.")
    log.info("Val: %d sequences (%d images). Pool: %d sequences (%d images).",
             len(val_seq_ids),
             sum(len(seqs[s]) for s in val_seq_ids),
             len(pool_seq_ids),
             sum(len(seqs[s]) for s in pool_seq_ids))

    # 3. Build held-out val dir
    val_dir = out_root / "val_holdout"
    n_val_imgs = build_val_holdout(train_jsonl, train_data_dir, val_seq_ids, val_dir)
    log.info("Built val holdout at %s (%d images)", val_dir, n_val_imgs)

    # 4. Grid
    if args.grid:
        grid = sorted(set(args.grid))
        grid = [g for g in grid if 1 <= g <= len(pool_seq_ids)]
    else:
        grid = make_geometric_grid(len(pool_seq_ids), args.grid_points, args.min_sequences)
    log.info("Grid (n_sequences): %s", grid)

    # 5. Manifest (saved up front; updated as runs complete)
    manifest = {
        "timestamp": ts,
        "train_dir": str(train_dir),
        "out_root": str(out_root),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "val_sequence_ids": sorted(val_seq_ids),
        "pool_sequence_ids_ordered": pool_seq_ids,
        "grid": grid,
        "train_overrides": args.train_overrides,
        "runs": [],
    }
    manifest_path = out_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    extra_overrides = list(args.train_overrides)

    # 6. Sweep — nested subsets so larger grid points are supersets of smaller ones
    runs_root = out_root / "runs"
    runs_root.mkdir(exist_ok=True)
    results: list[dict] = []
    results_path = out_root / "results.json"

    for n_seq in grid:
        chosen = pool_seq_ids[:n_seq]
        n_imgs = sum(len(seqs[s]) for s in chosen)
        run_dir = runs_root / f"n_seq={n_seq}"

        run_record = {
            "n_sequences": n_seq,
            "n_images": n_imgs,
            "sequence_ids": chosen,
            "run_dir": str(run_dir),
            "status": "running",
        }
        manifest["runs"].append(run_record)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        log.info("=" * 80)
        log.info("Run: n_sequences=%d  n_images=%d  -> %s", n_seq, n_imgs, run_dir)
        log.info("=" * 80)

        ok = run_training(
            python_bin=args.python,
            repo_root=repo_root,
            train_dir=train_dir,
            val_dir=val_dir,
            train_seq_ids=chosen,
            run_dir=run_dir,
            extra_overrides=extra_overrides,
        )

        if not ok:
            log.warning("Training failed for n_sequences=%d. Skipping.", n_seq)
            run_record["status"] = "failed"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            continue

        # Read final_metrics.json
        metrics_file = run_dir / "final_metrics.json"
        if not metrics_file.exists():
            log.warning("No final_metrics.json at %s. Skipping.", metrics_file)
            run_record["status"] = "no_metrics"
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            continue

        with open(metrics_file) as f:
            metrics = json.load(f)
        coco = metrics.get("best_coco_metrics") or {}

        result = {
            "n_sequences": n_seq,
            "n_images": n_imgs,
            "AP50": coco.get("AP50", 0.0),
            "AP": coco.get("AP", 0.0),
            "AP75": coco.get("AP75", 0.0),
            "best_val_loss": metrics.get("best_val_loss"),
            "best_epoch": metrics.get("best_epoch"),
            "completed_epochs": metrics.get("completed_epochs"),
            "run_dir": str(run_dir),
        }
        results.append(result)
        run_record["status"] = "ok"
        run_record["AP50"] = result["AP50"]

        # Incremental save — survive a mid-sweep crash
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        log.info("n_sequences=%d  n_images=%d  AP50=%.4f", n_seq, n_imgs, result["AP50"])

    # 7. Plot
    plot_path = out_root / "learning_curve.png"
    plot_curve(results, plot_path)

    log.info("=" * 80)
    log.info("Learning curve study complete.")
    log.info("  Output dir:    %s", out_root)
    log.info("  Successful:    %d / %d runs", len(results), len(grid))
    log.info("  Manifest:      %s", manifest_path)
    log.info("  Results JSON:  %s", results_path)
    log.info("  Plot:          %s", plot_path)


if __name__ == "__main__":
    main()
