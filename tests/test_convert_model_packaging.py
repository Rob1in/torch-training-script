"""Packaging tests for convert_model.sh.

Runs the real convert_model.sh against an input directory of run dirs and
checks that the onnx_model/ package (the directory that gets tarred and
uploaded to the Viam registry) contains all relevant files, with
--pytorch-metrics on and off.

The heavy Python entry points (convert_to_onnx.py, src/eval.py,
compare_metrics.py) are replaced with lightweight stubs that reproduce the
real scripts' output-file contract, so the tests exercise the shell script's
packaging logic without needing torch or a real checkpoint.

By default a synthetic runs directory is fabricated in tmp_path. Point
CONVERT_MODEL_TEST_RUNS_DIR at a real directory of run dirs (each containing
best_model.pth and .hydra/config.yaml) to run the same checks against it.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

CONVERT_TO_ONNX_STUB = """\
import argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--config', required=True)
p.add_argument('--output', required=True)
p.add_argument('--device', default='cpu')
p.add_argument('--image-input')
p.add_argument('--dataset-dir')
a = p.parse_args()

out = Path(a.output)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_bytes(b'ONNX-STUB')
(out.parent / 'labels.txt').write_text('triangle\\n')
"""

# Mimics eval.py's output contract: writes
# run_dir/eval_<dataset>_<checkpoint-stem>_<format>/<model_type>_metrics.json
EVAL_STUB = """\
import json
import sys
from pathlib import Path

args = dict(a.split('=', 1) for a in sys.argv[1:])
run_dir = Path(args['run_dir'])
ds = Path(args['dataset_dir'])
ckpt = Path(args['checkpoint_path'])
fmt = ckpt.suffix.lstrip('.')
out = run_dir / f"eval_{ds.name}_{ckpt.stem}_{fmt}"
out.mkdir(parents=True, exist_ok=True)
model_type = 'onnx' if fmt == 'onnx' else 'faster_rcnn'
metrics = {
    'AP': 0.5, 'AP50': 0.7, 'AP75': 0.4, 'AR100': 0.6,
    'checkpoint': str(ckpt),
    'is_onnx': fmt == 'onnx',
    'dataset': {'jsonl': str(ds / 'dataset.jsonl'), 'data_dir': str(ds / 'data')},
}
(out / f"{model_type}_metrics.json").write_text(json.dumps(metrics))
"""

COMPARE_METRICS_STUB = """\
import json
import sys

pytorch_path, onnx_path, out_path = sys.argv[1:4]
comparison = {
    'pytorch_checkpoint': json.load(open(pytorch_path)).get('checkpoint'),
    'onnx_checkpoint': json.load(open(onnx_path)).get('checkpoint'),
}
with open(out_path, 'w') as f:
    json.dump(comparison, f)
"""

ALWAYS_PACKAGED = ['model.onnx', 'labels.txt', 'config.yaml', 'conversion_summary.txt']


@pytest.fixture
def workspace(tmp_path):
    """Sandbox with the real convert_model.sh and stubbed Python entry points."""
    shutil.copy(REPO_ROOT / 'convert_model.sh', tmp_path / 'convert_model.sh')
    (tmp_path / 'convert_to_onnx.py').write_text(CONVERT_TO_ONNX_STUB)
    (tmp_path / 'compare_metrics.py').write_text(COMPARE_METRICS_STUB)
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / 'eval.py').write_text(EVAL_STUB)

    dataset = tmp_path / 'testset'
    (dataset / 'data').mkdir(parents=True)
    (dataset / 'dataset.jsonl').write_text('{}\n')
    return tmp_path


def make_run_dir(parent: Path, name: str) -> Path:
    run_dir = parent / name
    (run_dir / '.hydra').mkdir(parents=True)
    # workspace-absolute train_dir, as hydra records it at training time
    workspace = parent.parent
    (run_dir / '.hydra' / 'config.yaml').write_text(
        f'model:\n  name: faster_rcnn\ndataset:\n  train_dir: {workspace}/omnitrain\n')
    (run_dir / 'best_model.pth').write_bytes(b'PTH-STUB')
    return run_dir


@pytest.fixture
def runs_dir(workspace):
    """Input directory of runs: synthetic by default, or mirrored from a real one.

    External run dirs are mirrored into the sandbox (config copied, checkpoint
    symlinked) so the tests never write into the real runs directory —
    convert_model.sh creates its output package inside each run dir it's given.
    """
    runs = workspace / 'runs'
    external = os.environ.get('CONVERT_MODEL_TEST_RUNS_DIR')
    if not external:
        make_run_dir(runs, 'run_0')
        make_run_dir(runs, 'run_1')
        return runs

    mirrored = 0
    for src in sorted(Path(external).resolve().iterdir()):
        if not ((src / '.hydra' / 'config.yaml').is_file()
                and (src / 'best_model.pth').is_file()):
            continue
        dst = runs / src.name
        (dst / '.hydra').mkdir(parents=True)
        shutil.copy(src / '.hydra' / 'config.yaml', dst / '.hydra' / 'config.yaml')
        (dst / 'best_model.pth').symlink_to(src / 'best_model.pth')
        mirrored += 1
    assert mirrored, f'no run dirs with best_model.pth and .hydra/config.yaml in {external}'
    return runs


def make_pytorch_metrics(path: Path, run_dir: Path, dataset_name: str = 'omniteststitch') -> dict:
    """A faster_rcnn_metrics.json with absolute paths, as eval.py writes it."""
    metrics = {
        'AP': 0.282, 'AP50': 0.643, 'AP75': 0.191, 'AR100': 0.441,
        'checkpoint': str((run_dir / 'best_model.pth').resolve()),
        'is_onnx': False,
        'dataset': {
            'jsonl': f'/home/someone/{dataset_name}/dataset.jsonl',
            'data_dir': f'/home/someone/{dataset_name}/data',
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics))
    return metrics


def run_convert(workspace: Path, run_dir: Path, *extra_args: str) -> subprocess.CompletedProcess:
    cmd = ['bash', 'convert_model.sh', str(run_dir), '--dataset-dir', 'testset', *extra_args]
    return subprocess.run(cmd, cwd=workspace, capture_output=True, text=True)


def list_run_dirs(runs_dir: Path) -> list:
    run_dirs = sorted(d for d in runs_dir.iterdir() if (d / '.hydra' / 'config.yaml').is_file())
    assert run_dirs, f'no run dirs found in {runs_dir}'
    return run_dirs


def test_package_without_pytorch_metrics_flag(workspace, runs_dir):
    for run_dir in list_run_dirs(runs_dir):
        result = run_convert(workspace, run_dir)
        assert result.returncode == 0, result.stderr + result.stdout

        package = run_dir / 'onnx_model'
        for name in ALWAYS_PACKAGED:
            assert (package / name).is_file(), f'{name} missing from {package}'
        # Config must be the training config, copied verbatim
        source = (run_dir / '.hydra' / 'config.yaml').read_text()
        packaged = (package / 'config.yaml').read_text()
        assert packaged == source
        # No metrics without the flag
        assert not (package / 'pytorch_metrics.json').exists()


def test_package_with_pytorch_metrics_flag(workspace, runs_dir):
    for run_dir in list_run_dirs(runs_dir):
        metrics_path = workspace / f'{run_dir.name}_faster_rcnn_metrics.json'
        original = make_pytorch_metrics(metrics_path, run_dir)

        result = run_convert(workspace, run_dir, '--pytorch-metrics', str(metrics_path))
        assert result.returncode == 0, result.stderr + result.stdout

        package = run_dir / 'onnx_model'
        for name in ALWAYS_PACKAGED + ['pytorch_metrics.json']:
            assert (package / name).is_file(), f'{name} missing from {package}'

        packaged = json.loads((package / 'pytorch_metrics.json').read_text())
        # Metrics are copied verbatim
        assert packaged == original


def test_full_package_with_eval_and_comparison(workspace, runs_dir):
    """--evaluate-converted-model + --pytorch-metrics -> the complete uploadable
    package. The provided metrics feed both packaging and the comparison (no
    prior eval dir exists, so the comparison can only have used the flag)."""
    run_dir = list_run_dirs(runs_dir)[0]
    metrics_path = workspace / 'pytorch_metrics.json'
    make_pytorch_metrics(metrics_path, run_dir, dataset_name='testset')

    result = run_convert(workspace, run_dir, '--evaluate-converted-model',
                         '--pytorch-metrics', str(metrics_path))
    assert result.returncode == 0, result.stderr + result.stdout

    package = run_dir / 'onnx_model'
    expected = ALWAYS_PACKAGED + ['pytorch_metrics.json', 'comparison.json']
    for name in expected:
        assert (package / name).is_file(), f'{name} missing from {package}'
    # The ONNX eval itself lands outside the package
    assert (run_dir / 'eval_testset_model_onnx' / 'onnx_metrics.json').is_file()
    # Nothing unexpected ships in the package
    assert sorted(p.name for p in package.iterdir()) == sorted(expected)


def test_comparison_discovers_prior_pytorch_eval(workspace, runs_dir):
    """Without --pytorch-metrics, the comparison falls back to the conventional
    prior-eval location: eval_<dataset>_<checkpoint>_pth/faster_rcnn_metrics.json."""
    run_dir = list_run_dirs(runs_dir)[0]
    make_pytorch_metrics(
        run_dir / 'eval_testset_best_model_pth' / 'faster_rcnn_metrics.json',
        run_dir, dataset_name='testset')

    result = run_convert(workspace, run_dir, '--evaluate-converted-model')
    assert result.returncode == 0, result.stderr + result.stdout

    package = run_dir / 'onnx_model'
    assert (package / 'comparison.json').is_file()
    # Discovery feeds the comparison only; nothing extra is packaged
    assert not (package / 'pytorch_metrics.json').exists()


def test_comparison_falls_back_on_dataset_mismatch(workspace, runs_dir):
    """--pytorch-metrics from a different dataset than --dataset-dir: the
    comparison falls back to the conventional prior eval; the provided metrics
    still ship in the package."""
    run_dir = list_run_dirs(runs_dir)[0]
    make_pytorch_metrics(
        run_dir / 'eval_testset_best_model_pth' / 'faster_rcnn_metrics.json',
        run_dir, dataset_name='testset')
    metrics_path = workspace / 'benchmark_metrics.json'
    make_pytorch_metrics(metrics_path, run_dir)  # dataset omniteststitch

    result = run_convert(workspace, run_dir, '--evaluate-converted-model',
                         '--pytorch-metrics', str(metrics_path))
    assert result.returncode == 0, result.stderr + result.stdout
    assert 'Falling back to' in result.stdout

    package = run_dir / 'onnx_model'
    assert (package / 'comparison.json').is_file()
    assert (package / 'pytorch_metrics.json').is_file()


def test_comparison_skipped_on_dataset_mismatch_without_fallback(workspace, runs_dir):
    """--pytorch-metrics from a different dataset and no conventional prior
    eval: the comparison is refused, but the run still succeeds and the
    metrics still ship."""
    run_dir = list_run_dirs(runs_dir)[0]
    metrics_path = workspace / 'benchmark_metrics.json'
    make_pytorch_metrics(metrics_path, run_dir)  # dataset omniteststitch

    result = run_convert(workspace, run_dir, '--evaluate-converted-model',
                         '--pytorch-metrics', str(metrics_path))
    assert result.returncode == 0, result.stderr + result.stdout
    assert 'Skipping comparison' in result.stdout

    package = run_dir / 'onnx_model'
    assert not (package / 'comparison.json').exists()
    assert (package / 'pytorch_metrics.json').is_file()


def test_rerun_without_flags_drops_stale_artifacts(workspace, runs_dir):
    """A re-run rebuilds the package from scratch: conditional artifacts from a
    previous invocation must not survive into a run that didn't request them."""
    run_dir = list_run_dirs(runs_dir)[0]
    metrics_path = workspace / 'benchmark_metrics.json'
    make_pytorch_metrics(metrics_path, run_dir)

    result = run_convert(workspace, run_dir, '--pytorch-metrics', str(metrics_path))
    assert result.returncode == 0, result.stderr + result.stdout
    package = run_dir / 'onnx_model'
    assert (package / 'pytorch_metrics.json').is_file()

    result = run_convert(workspace, run_dir)
    assert result.returncode == 0, result.stderr + result.stdout
    assert not (package / 'pytorch_metrics.json').exists()
    assert sorted(p.name for p in package.iterdir()) == sorted(ALWAYS_PACKAGED)


def test_missing_pytorch_metrics_file_fails_fast(workspace, runs_dir):
    run_dir = list_run_dirs(runs_dir)[0]
    result = run_convert(workspace, run_dir,
                         '--pytorch-metrics', str(workspace / 'does_not_exist.json'))
    assert result.returncode != 0
    assert 'not found' in result.stdout + result.stderr
    # Failed validation must not leave a package behind
    assert not (run_dir / 'onnx_model').exists()

    # ...and must not wipe an existing package either
    result = run_convert(workspace, run_dir)
    assert result.returncode == 0, result.stderr + result.stdout
    result = run_convert(workspace, run_dir,
                         '--pytorch-metrics', str(workspace / 'does_not_exist.json'))
    assert result.returncode != 0
    assert (run_dir / 'onnx_model' / 'model.onnx').is_file()
