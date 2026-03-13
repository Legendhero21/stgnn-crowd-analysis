"""
Compare All Evaluation Results — Print Report-Ready Tables

Reads JSON outputs from run_eval_batch.py and prints comparison tables
that can be directly copy-pasted into your thesis/report.

Usage:
  python scripts/compare_all.py
  python scripts/compare_all.py --results-dir outputs/eval_results
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "eval_results")


def load_results(results_dir: str) -> tuple:
    """Load pipeline and YOLO validation JSONs."""
    pipeline_results = []
    yolo_results = []

    for jf in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(jf) as f:
            data = json.load(f)

        basename = os.path.basename(jf)
        if basename.startswith("yolo_val_"):
            yolo_results.append(data)
        else:
            pipeline_results.append(data)

    return pipeline_results, yolo_results


def print_yolo_table(yolo_results: list) -> None:
    """Print YOLO validation comparison table."""
    if not yolo_results:
        print("  No YOLO validation results found.")
        return

    print(f"  {'Model':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'mAP@50':>10} {'mAP@50-95':>10}")
    print("  " + "-" * 80)

    for r in yolo_results:
        print(
            f"  {r.get('model', '?'):<30} "
            f"{r.get('precision', 0):.4f}     "
            f"{r.get('recall', 0):.4f}     "
            f"{r.get('f1', 0):.4f}     "
            f"{r.get('mAP50', 0):.4f}     "
            f"{r.get('mAP50_95', 0):.4f}"
        )


def print_pipeline_table(pipeline_results: list) -> None:
    """Print full pipeline comparison table."""
    if not pipeline_results:
        print("  No pipeline results found.")
        return

    # Sort by tag
    pipeline_results.sort(key=lambda x: x.get("tag", ""))

    print(f"  {'Tag':<22} {'Model':<18} {'Video':<22} {'AvgPers':>8} {'AvgAnom':>10} {'%NOR':>6} {'%UNS':>6} {'%STP':>6} {'FPS':>7} {'InfMs':>7}")
    print("  " + "-" * 125)

    for r in pipeline_results:
        print(
            f"  {r.get('tag', '?'):<22} "
            f"{r.get('model_type', '?'):<18} "
            f"{r.get('video', '?'):<22} "
            f"{r.get('avg_persons', 0):>8.1f} "
            f"{r.get('avg_anomaly', 0):>10.6f} "
            f"{r.get('pct_normal', 0):>6.1f} "
            f"{r.get('pct_unstable', 0):>6.1f} "
            f"{r.get('pct_stampede', 0):>6.1f} "
            f"{r.get('avg_fps', 0):>7.1f} "
            f"{r.get('yolo_avg_inference_ms', 0):>7.1f}"
        )


def print_resource_table(pipeline_results: list) -> None:
    """Print resource usage comparison."""
    if not pipeline_results:
        return

    print(f"  {'Tag':<22} {'Fed':>4} {'Frames':>8} {'Time(s)':>8} {'FPS':>7} {'InfMs':>7} {'MemMB':>7} {'GPU':>6}")
    print("  " + "-" * 80)

    for r in pipeline_results:
        print(
            f"  {r.get('tag', '?'):<22} "
            f"{'Y' if r.get('federated') else 'N':>4} "
            f"{r.get('total_frames', 0):>8} "
            f"{r.get('total_time_s', 0):>8.1f} "
            f"{r.get('avg_fps', 0):>7.1f} "
            f"{r.get('yolo_avg_inference_ms', 0):>7.1f} "
            f"{r.get('mem_used_mb', 0):>7.1f} "
            f"{r.get('gpu_device', '?'):>6}"
        )


def print_crowd_metrics_table(pipeline_results: list) -> None:
    """Print crowd analysis metrics."""
    if not pipeline_results:
        return

    print(f"  {'Tag':<22} {'AvgDensity':>12} {'AvgEntropy':>12} {'AvgSpeed':>12} {'MaxAnom':>12} {'StdAnom':>12}")
    print("  " + "-" * 85)

    for r in pipeline_results:
        print(
            f"  {r.get('tag', '?'):<22} "
            f"{r.get('avg_density', 0):>12.4f} "
            f"{r.get('avg_entropy', 0):>12.4f} "
            f"{r.get('avg_speed', 0):>12.4f} "
            f"{r.get('max_anomaly', 0):>12.6f} "
            f"{r.get('std_anomaly', 0):>12.6f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation results")
    parser.add_argument(
        "--results-dir", type=str, default=DEFAULT_RESULTS_DIR,
        help=f"Directory with JSON results (default: {DEFAULT_RESULTS_DIR})"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"Results directory not found: {args.results_dir}")
        sys.exit(1)

    pipeline_results, yolo_results = load_results(args.results_dir)

    print("\n")
    print("=" * 130)
    print("  COMPREHENSIVE MODEL EVALUATION REPORT")
    print("=" * 130)

    print("\n  TABLE 1: YOLO Detection Metrics (Labeled Dataset Validation)")
    print("  " + "=" * 80)
    print_yolo_table(yolo_results)

    print("\n  TABLE 2: Full Pipeline — Model × Video Comparison")
    print("  " + "=" * 125)
    print_pipeline_table(pipeline_results)

    print("\n  TABLE 3: Crowd Analysis Metrics")
    print("  " + "=" * 85)
    print_crowd_metrics_table(pipeline_results)

    print("\n  TABLE 4: Resource Usage & Performance")
    print("  " + "=" * 80)
    print_resource_table(pipeline_results)

    # Federated comparison
    fed = [r for r in pipeline_results if r.get("federated")]
    nonfed = [r for r in pipeline_results if not r.get("federated")]
    if fed and nonfed:
        print("\n  TABLE 5: Federated vs Non-Federated Comparison")
        print("  " + "=" * 80)
        print_resource_table(fed + nonfed)

    print("\n" + "=" * 130)
    print(f"  Total results: {len(yolo_results)} YOLO validations, {len(pipeline_results)} pipeline runs")
    print("=" * 130)
    print("")


if __name__ == "__main__":
    main()
