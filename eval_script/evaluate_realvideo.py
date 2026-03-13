"""
Real-Video Pipeline Evaluation
================================
Reads the per-frame JSON produced by run_pipeline_realtime.py and computes:
  - State distribution (NORMAL / UNSTABLE / STAMPEDE) with percentages
  - Anomaly score statistics (mean, std, p50, p90, p95, p99, max)
  - Temporal anomaly plot saved as PNG
  - Rolling state timeline saved as PNG
  - Full text report saved to outputs/evaluation/

Usage:
    python scripts/evaluate_realvideo.py
    python scripts/evaluate_realvideo.py --json path/to/realvideo_metrics_*.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "evaluation")


def find_latest_metrics(results_dir: str) -> str | None:
    """Return the most recently modified realvideo_metrics_*.json file."""
    d = Path(results_dir)
    matches = sorted(d.glob("realvideo_metrics_*.json"), key=lambda p: p.stat().st_mtime)
    return str(matches[-1]) if matches else None


def evaluate(json_path: str, output_dir: str) -> None:
    with open(json_path, "r") as f:
        data = json.load(f)

    video_name = data.get("video_name", "unknown")
    total_frames = data.get("total_frames", 0)
    evaluated_frames = data.get("evaluated_frames", 0)
    state_counts = data.get("state_counts", {})
    frames = data.get("frames", [])

    if not frames:
        print("[ERROR] No frame data found in metrics JSON.")
        return

    # ----------------------------------------------------------------
    # Per-frame arrays
    # ----------------------------------------------------------------
    anomaly_scores = np.array([fr["anomaly_score"] for fr in frames], dtype=np.float64)
    person_counts  = np.array([fr["person_count"]  for fr in frames], dtype=np.float64)
    densities      = np.array([fr["mean_density"]   for fr in frames], dtype=np.float64)
    entropies      = np.array([fr["motion_entropy"] for fr in frames], dtype=np.float64)
    states         = [fr["state"] for fr in frames]
    frame_indices  = [fr["frame"] for fr in frames]

    # ----------------------------------------------------------------
    # Anomaly score stats
    # ----------------------------------------------------------------
    stats = {
        "mean":  float(np.mean(anomaly_scores)),
        "std":   float(np.std(anomaly_scores)),
        "min":   float(np.min(anomaly_scores)),
        "p50":   float(np.percentile(anomaly_scores, 50)),
        "p90":   float(np.percentile(anomaly_scores, 90)),
        "p95":   float(np.percentile(anomaly_scores, 95)),
        "p99":   float(np.percentile(anomaly_scores, 99)),
        "max":   float(np.max(anomaly_scores)),
    }

    # Fraction of frames the model was "active" (had persons, non-zero anomaly)
    active_frames = int(np.sum(anomaly_scores > 0))

    # ----------------------------------------------------------------
    # State distribution
    # ----------------------------------------------------------------
    total_ev = len(frames)
    state_pct = {
        s: round(100.0 * state_counts.get(s, 0) / total_ev, 2) if total_ev else 0.0
        for s in ("NORMAL", "UNSTABLE", "STAMPEDE")
    }

    # Longest consecutive STAMPEDE / UNSTABLE run
    def longest_run(target: str) -> int:
        best = cur = 0
        for s in states:
            cur = cur + 1 if s == target else 0
            best = max(best, cur)
        return best

    longest_stampede  = longest_run("STAMPEDE")
    longest_unstable  = longest_run("UNSTABLE")

    # ----------------------------------------------------------------
    # Console + text report
    # ----------------------------------------------------------------
    sep  = "=" * 62
    dash = "-" * 62

    lines = [
        sep,
        "  REAL-VIDEO PIPELINE EVALUATION REPORT",
        sep,
        f"  Video       : {video_name}",
        f"  Source JSON : {json_path}",
        f"  Timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Total frames: {total_frames}  |  Evaluated: {evaluated_frames}",
        f"  Active frames (anomaly > 0): {active_frames}",
        "",
        "STATE DISTRIBUTION",
        dash,
        f"  NORMAL    : {state_counts.get('NORMAL',   0):>5d} frames  ({state_pct['NORMAL']:.1f}%)",
        f"  UNSTABLE  : {state_counts.get('UNSTABLE', 0):>5d} frames  ({state_pct['UNSTABLE']:.1f}%)",
        f"  STAMPEDE  : {state_counts.get('STAMPEDE', 0):>5d} frames  ({state_pct['STAMPEDE']:.1f}%)",
        f"  Longest consecutive UNSTABLE run : {longest_unstable} frames",
        f"  Longest consecutive STAMPEDE run : {longest_stampede} frames",
        "",
        "ANOMALY SCORE STATISTICS",
        dash,
        f"  Mean   : {stats['mean']:.6f}",
        f"  Std    : {stats['std']:.6f}",
        f"  Median : {stats['p50']:.6f}",
        f"  P90    : {stats['p90']:.6f}",
        f"  P95    : {stats['p95']:.6f}",
        f"  P99    : {stats['p99']:.6f}",
        f"  Max    : {stats['max']:.6f}",
        "",
        "CROWD METRICS (mean across frames)",
        dash,
        f"  Avg person count : {float(np.mean(person_counts)):.2f}",
        f"  Avg mean density : {float(np.mean(densities)):.4f}",
        f"  Avg mot. entropy : {float(np.mean(entropies)):.4f}",
        sep,
    ]

    report_text = "\n".join(lines)
    print("\n" + report_text + "\n")

    # ----------------------------------------------------------------
    # Save text report
    # ----------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"realvideo_report_{video_name}_{ts}.txt")
    with open(report_path, "w") as f:
        f.write(report_text + "\n")
    print(f"[OK] Text report → {report_path}")

    # ----------------------------------------------------------------
    # Save plots
    # ----------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        color_map = {"NORMAL": "green", "UNSTABLE": "orange", "STAMPEDE": "red"}

        # -- Plot 1: Anomaly score over time --
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(frame_indices, anomaly_scores, lw=0.7, color="steelblue", label="Anomaly score")
        ax.axhline(stats["p95"], color="orange", lw=1, ls="--", label=f"P95 = {stats['p95']:.4f}")
        ax.axhline(stats["p99"], color="red",    lw=1, ls="--", label=f"P99 = {stats['p99']:.4f}")
        ax.set_title(f"Anomaly Score Timeline — {video_name}", fontsize=13)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Anomaly Score (MSE)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plot1_path = os.path.join(output_dir, f"realvideo_anomaly_timeline_{video_name}_{ts}.png")
        fig.savefig(plot1_path, dpi=150)
        plt.close(fig)
        print(f"[OK] Anomaly timeline plot → {plot1_path}")

        # -- Plot 2: State timeline (colour-coded strip) --
        state_num = {"NORMAL": 0, "UNSTABLE": 1, "STAMPEDE": 2}
        state_vals = np.array([state_num[s] for s in states])
        colors     = [color_map[s] for s in states]

        fig, ax = plt.subplots(figsize=(14, 2))
        ax.bar(frame_indices, np.ones(len(frame_indices)), color=colors, width=1.0, align="edge")
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor="green",  label="NORMAL"),
                           Patch(facecolor="orange", label="UNSTABLE"),
                           Patch(facecolor="red",    label="STAMPEDE")]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
        ax.set_title(f"State Timeline — {video_name}", fontsize=12)
        ax.set_xlabel("Frame")
        ax.set_yticks([])
        ax.set_xlim(frame_indices[0], frame_indices[-1])
        plt.tight_layout()
        plot2_path = os.path.join(output_dir, f"realvideo_state_timeline_{video_name}_{ts}.png")
        fig.savefig(plot2_path, dpi=150)
        plt.close(fig)
        print(f"[OK] State timeline plot → {plot2_path}")

        # -- Plot 3: Person count over time --
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.fill_between(frame_indices, person_counts, alpha=0.4, color="teal")
        ax.plot(frame_indices, person_counts, lw=0.8, color="teal")
        ax.set_title(f"Person Count Over Time — {video_name}", fontsize=12)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Detected persons")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plot3_path = os.path.join(output_dir, f"realvideo_person_count_{video_name}_{ts}.png")
        fig.savefig(plot3_path, dpi=150)
        plt.close(fig)
        print(f"[OK] Person count plot → {plot3_path}")

    except Exception as exc:
        print(f"[WARN] Could not generate plots: {exc}")

    # ----------------------------------------------------------------
    # Save evaluation JSON summary (no per-frame data, compact)
    # ----------------------------------------------------------------
    eval_json = {
        "video_name": video_name,
        "timestamp": ts,
        "total_frames": total_frames,
        "evaluated_frames": evaluated_frames,
        "active_frames": active_frames,
        "state_counts": state_counts,
        "state_percentages": state_pct,
        "longest_runs": {
            "UNSTABLE": longest_unstable,
            "STAMPEDE": longest_stampede,
        },
        "anomaly_score_stats": stats,
        "crowd": {
            "avg_person_count": round(float(np.mean(person_counts)), 2),
            "avg_mean_density": round(float(np.mean(densities)), 4),
            "avg_motion_entropy": round(float(np.mean(entropies)), 4),
        },
    }
    eval_json_path = os.path.join(output_dir, f"realvideo_eval_{video_name}_{ts}.json")
    with open(eval_json_path, "w") as f:
        json.dump(eval_json, f, indent=2)
    print(f"[OK] Evaluation JSON → {eval_json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate real-video pipeline output from run_pipeline_realtime.py"
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Path to realvideo_metrics_*.json produced by the pipeline. "
             "If omitted, the latest file in outputs/pipeline_results/ is used.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save evaluation reports and plots.",
    )
    args = parser.parse_args()

    json_path = args.json
    if json_path is None:
        results_dir = str(PROJECT_ROOT / "outputs" / "pipeline_results")
        json_path = find_latest_metrics(results_dir)
        if json_path is None:
            print(
                "[ERROR] No realvideo_metrics_*.json found in outputs/pipeline_results/.\n"
                "Run `python src/run_pipeline_realtime.py` first, then re-run this script."
            )
            sys.exit(1)
        print(f"[INFO] Using latest metrics file: {json_path}")

    evaluate(json_path, args.output_dir)


if __name__ == "__main__":
    main()
