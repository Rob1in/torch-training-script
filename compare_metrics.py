#!/usr/bin/env python3
"""
Compare PyTorch and ONNX model evaluation metrics.

This script compares evaluation results from PyTorch and ONNX models,
checking for dataset consistency and computing metric differences.
"""
import argparse
import json
import sys
from pathlib import Path


def load_metrics(metrics_path: Path) -> dict:
    """Load metrics from a JSON file."""
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    
    with open(metrics_path, 'r') as f:
        return json.load(f)


def check_dataset_consistency(pytorch_metrics: dict, onnx_metrics: dict):
    """Check if both evaluations used the same dataset."""
    pytorch_dataset = pytorch_metrics.get('dataset', {})
    onnx_dataset = onnx_metrics.get('dataset', {})
    
    dataset_match = (
        pytorch_dataset.get('jsonl') == onnx_dataset.get('jsonl') and
        pytorch_dataset.get('data_dir') == onnx_dataset.get('data_dir')
    )
    
    return dataset_match, pytorch_dataset, onnx_dataset


def compare_metrics(pytorch_metrics: dict, onnx_metrics: dict, metrics_to_compare: list) -> dict:
    """Compare metrics between PyTorch and ONNX models."""
    comparison = {}
    
    for metric in metrics_to_compare:
        pytorch_val = pytorch_metrics.get(metric, 0.0)
        onnx_val = onnx_metrics.get(metric, 0.0)
        diff = onnx_val - pytorch_val
        
        comparison[metric] = {
            'pytorch': pytorch_val,
            'onnx': onnx_val,
            'difference': diff
        }
    
    return comparison


def print_comparison(pytorch_metrics: dict, onnx_metrics: dict, comparison: dict):
    """Print formatted comparison results."""
    # Check dataset consistency
    dataset_match, pytorch_dataset, onnx_dataset = check_dataset_consistency(
        pytorch_metrics, onnx_metrics
    )
    
    if not dataset_match:
        print("⚠️  WARNING: PyTorch and ONNX evaluations used different datasets!")
        print(f"  PyTorch: {pytorch_dataset}")
        print(f"  ONNX: {onnx_dataset}")
    else:
        print("✓ Both models evaluated on the same dataset")
    
    print()
    print("Metric Comparison:")
    print("-" * 60)
    
    for metric, values in comparison.items():
        pytorch_val = values['pytorch']
        onnx_val = values['onnx']
        diff = values['difference']
        diff_pct = (diff / pytorch_val * 100) if pytorch_val > 0 else 0
        
        print(f"{metric:8s}: PyTorch={pytorch_val:.4f}  ONNX={onnx_val:.4f}  "
              f"Diff={diff:+.4f} ({diff_pct:+.2f}%)")
    
    # Check if differences are acceptable
    max_diff = max(abs(v['difference']) for v in comparison.values())
    print()
    
    if max_diff < 0.001:
        print("✅ EXCELLENT: ONNX and PyTorch models produce nearly identical results!")
    elif max_diff < 0.01:
        print("✅ GOOD: ONNX and PyTorch models produce very similar results")
    else:
        print("⚠️  WARNING: Significant differences detected between models")
        print("   This could be due to:")
        print("   - Different evaluation datasets")
        print("   - Numerical precision differences")
        print("   - ONNX export issues")


def main():
    parser = argparse.ArgumentParser(
        description='Compare PyTorch and ONNX model evaluation metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s pytorch_metrics.json onnx_metrics.json comparison.json
  %(prog)s --pytorch-metrics pytorch.json --onnx-metrics onnx.json --output comparison.json
        """
    )
    parser.add_argument(
        'pytorch_metrics',
        nargs='?',
        help='Path to PyTorch model metrics JSON file'
    )
    parser.add_argument(
        'onnx_metrics',
        nargs='?',
        help='Path to ONNX model metrics JSON file'
    )
    parser.add_argument(
        'output',
        nargs='?',
        help='Path to save comparison JSON file'
    )
    parser.add_argument(
        '--pytorch-metrics',
        dest='pytorch_metrics_opt',
        help='Path to PyTorch model metrics JSON file (alternative to positional)'
    )
    parser.add_argument(
        '--onnx-metrics',
        dest='onnx_metrics_opt',
        help='Path to ONNX model metrics JSON file (alternative to positional)'
    )
    parser.add_argument(
        '--output', '-o',
        dest='output_opt',
        help='Path to save comparison JSON file (alternative to positional)'
    )
    
    args = parser.parse_args()
    
    # Handle both positional and optional arguments
    pytorch_path = args.pytorch_metrics_opt or args.pytorch_metrics
    onnx_path = args.onnx_metrics_opt or args.onnx_metrics
    output_path = args.output_opt or args.output
    
    if not pytorch_path or not onnx_path:
        parser.error("Both PyTorch and ONNX metrics paths are required")
    
    if not output_path:
        parser.error("Output path is required")
    
    pytorch_path = Path(pytorch_path)
    onnx_path = Path(onnx_path)
    output_path = Path(output_path)
    
    # Load metrics
    try:
        pytorch_metrics = load_metrics(pytorch_path)
        onnx_metrics = load_metrics(onnx_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Compare metrics
    metrics_to_compare = ['AP', 'AP50', 'AP75', 'AR100']
    comparison_dict = compare_metrics(pytorch_metrics, onnx_metrics, metrics_to_compare)
    
    # Check dataset consistency
    dataset_match, pytorch_dataset, onnx_dataset = check_dataset_consistency(
        pytorch_metrics, onnx_metrics
    )
    
    # Create comparison summary
    comparison = {
        'pytorch_checkpoint': pytorch_metrics.get('checkpoint', 'unknown'),
        'onnx_checkpoint': onnx_metrics.get('checkpoint', 'unknown'),
        'dataset_match': dataset_match,
        'dataset': onnx_dataset,
        'metrics': comparison_dict
    }
    
    # Print comparison
    print_comparison(pytorch_metrics, onnx_metrics, comparison_dict)
    
    # Save comparison
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=4)
    
    print()
    print(f"✓ Comparison saved to: {output_path}")


if __name__ == '__main__':
    main()
