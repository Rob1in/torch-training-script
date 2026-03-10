#!/usr/bin/env python3
"""
Evaluate all runs in a sweep directory.

Discovers run subdirectories (any directory with .hydra/config.yaml + best_model.pth),
calls eval.py for each one, and prints a ranked summary table.

Usage (same arguments as eval.py, but sweep_dir instead of run_dir):
    python src/eval_sweep.py sweep_dir=outputs/2026-03-05/21-37-10 dataset_dir=my_test_data
    python src/eval_sweep.py sweep_dir=outputs/2026-03-05/21-37-10 dataset_dir=my_test_data evaluation.visualize=true
"""
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def discover_run_dirs(sweep_dir: Path) -> list[Path]:
    """Find subdirectories that contain both .hydra/config.yaml and best_model.pth."""
    run_dirs = []
    for child in sorted(sweep_dir.iterdir()):
        if not child.is_dir():
            continue
        has_config = (child / ".hydra" / "config.yaml").exists()
        has_checkpoint = (child / "best_model.pth").exists()
        if has_config and has_checkpoint:
            run_dirs.append(child)
        elif has_config and not has_checkpoint:
            log.warning(f"Skipping {child.name}: has .hydra/config.yaml but no best_model.pth (training incomplete?)")
    return run_dirs


def find_metrics_json(run_dir: Path) -> Path | None:
    """Find the first *_metrics.json inside an eval_* subdirectory of run_dir."""
    for eval_dir in sorted(run_dir.iterdir()):
        if not eval_dir.is_dir() or not eval_dir.name.startswith("eval_"):
            continue
        for f in sorted(eval_dir.iterdir()):
            if f.name.endswith("_metrics.json"):
                return f
    return None


def collect_summary(run_dirs: list[Path]) -> list[dict]:
    """Read metrics JSONs from evaluated runs and build summary rows."""
    rows = []
    for run_dir in run_dirs:
        metrics_path = find_metrics_json(run_dir)
        if metrics_path is None:
            log.warning(f"No metrics found for {run_dir.name}, skipping from summary")
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)

        pr = metrics.get("precision_recall", {}).get("overall", {})
        rows.append({
            "run": run_dir.name,
            "AP": metrics.get("AP", 0.0),
            "AP50": metrics.get("AP50", 0.0),
            "AP75": metrics.get("AP75", 0.0),
            "precision": pr.get("precision", 0.0),
            "recall": pr.get("recall", 0.0),
            "f1": pr.get("f1", 0.0),
            "metrics_file": str(metrics_path),
        })
    rows.sort(key=lambda r: r["AP"], reverse=True)
    return rows


def print_summary(rows: list[dict]):
    if not rows:
        log.warning("No metrics to summarize.")
        return

    name_w = max(len(r["run"]) for r in rows)
    name_w = max(name_w, 3)

    header = (
        f"  {'#':>3}  {'Run':<{name_w}}  {'AP':>7}  {'AP50':>7}  {'AP75':>7}"
        f"  {'Prec':>7}  {'Rec':>7}  {'F1':>7}"
    )
    sep = f"  {'---':>3}  {'-' * name_w}  {'-------':>7}  {'-------':>7}  {'-------':>7}  {'-------':>7}  {'-------':>7}  {'-------':>7}"

    log.info("=" * len(header))
    log.info("  Sweep Evaluation Summary (ranked by AP)")
    log.info("=" * len(header))
    log.info(header)
    log.info(sep)
    for i, r in enumerate(rows, 1):
        log.info(
            f"  {i:>3}  {r['run']:<{name_w}}  {r['AP']:7.4f}  {r['AP50']:7.4f}  {r['AP75']:7.4f}"
            f"  {r['precision']:7.4f}  {r['recall']:7.4f}  {r['f1']:7.4f}"
        )
    log.info("=" * len(header))


def main():
    # Parse sweep_dir from argv; forward everything else to eval.py
    sweep_dir_str = None
    forwarded_args = []
    for arg in sys.argv[1:]:
        if arg.startswith("sweep_dir="):
            sweep_dir_str = arg.split("=", 1)[1]
        else:
            forwarded_args.append(arg)

    if sweep_dir_str is None:
        print(
            "Usage: python src/eval_sweep.py sweep_dir=<sweep_dir> dataset_dir=<dir> [eval.py args...]\n"
            "Example: python src/eval_sweep.py sweep_dir=outputs/2026-03-05/21-37-10 dataset_dir=my_test_data",
            file=sys.stderr,
        )
        sys.exit(1)

    sweep_dir = Path(sweep_dir_str)
    if not sweep_dir.is_absolute():
        sweep_dir = Path.cwd() / sweep_dir

    if not sweep_dir.exists():
        log.error(f"Sweep directory not found: {sweep_dir}")
        sys.exit(1)

    run_dirs = discover_run_dirs(sweep_dir)
    if not run_dirs:
        log.error(f"No valid run directories found in {sweep_dir}")
        log.error("A valid run directory must contain .hydra/config.yaml and best_model.pth")
        sys.exit(1)

    log.info(f"Found {len(run_dirs)} runs in {sweep_dir}")
    for rd in run_dirs:
        log.info(f"  {rd.name}")

    # Evaluate each run
    eval_script = Path(__file__).resolve().parent / "eval.py"
    failed = []
    for i, run_dir in enumerate(run_dirs, 1):
        log.info("=" * 80)
        log.info(f"[{i}/{len(run_dirs)}] Evaluating: {run_dir.name}")
        log.info("=" * 80)

        cmd = [
            sys.executable, str(eval_script),
            f"run_dir={run_dir}",
            *forwarded_args,
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            log.error(f"Evaluation FAILED for {run_dir.name} (exit code {result.returncode})")
            failed.append(run_dir.name)

    # Summary
    log.info("")
    log.info(f"Evaluation complete: {len(run_dirs) - len(failed)}/{len(run_dirs)} succeeded")
    if failed:
        log.warning(f"Failed runs: {', '.join(failed)}")

    rows = collect_summary(run_dirs)
    print_summary(rows)

    summary_path = sweep_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(rows, f, indent=2)
    log.info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
