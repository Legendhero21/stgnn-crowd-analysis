"""
End-to-End Pipeline Test
=========================
Tests the entire crowd anomaly detection pipeline:
  TemporalGraphBuffer → STGNN (PyTorch) → CrowdMetrics → StampedeAlert

Reports comprehensive metrics including:
  - Regression: MSE, RMSE, MAE, R², MAPE
  - Classification: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Cohen's Kappa
  - Per-class breakdown and confusion matrix
  - Pipeline latency benchmarks

Usage:
    python scripts/test_pipeline_e2e.py
    python scripts/test_pipeline_e2e.py --n-samples 500
    python scripts/test_pipeline_e2e.py --checkpoint path/to/model.pth
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("PipelineE2E")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT = str(
    PROJECT_ROOT / "outputs" / "evaluation" / "stgnn_final_frozen.pth"
)
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "evaluation")
TEMPORAL_WINDOW = 5

STGNN_CONFIG = {
    "in_channels": 5,
    "hidden_channels": 64,
    "out_channels": 2,
    "num_layers": 3,
    "dropout": 0.1,
    "kernel_size": 3,
}


# ============================================================================
# Pretty table
# ============================================================================
def _print_table(title: str, rows: List[Tuple[str, str]]) -> str:
    col_w = max(len(r[0]) for r in rows) + 2
    val_w = max(len(str(r[1])) for r in rows) + 2
    width = col_w + val_w + 3
    lines: List[str] = []
    lines.append("=" * width)
    lines.append(f"  {title}")
    lines.append("-" * width)
    for label, value in rows:
        lines.append(f"  {label:<{col_w}} {value:>{val_w}}")
    lines.append("=" * width)
    return "\n".join(lines)


# ============================================================================
# Synthetic scene generator
# ============================================================================
def generate_crowd_scenes(
    n_scenes: int = 200,
    temporal_window: int = 5,
    n_nodes: int = 10,
    n_features: int = 5,
    anomaly_ratio: float = 0.25,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate crowd scenes — each is a sequence of raw graph dicts
    (simulating what RealtimeGraphBuilder would produce frame-by-frame).

    Returns list of dicts with:
        frames   : List[Dict]  — T graph dicts with 'x' and 'edge_index'
        target   : np.ndarray [N, 2]
        label    : str
    """
    rng = np.random.RandomState(seed)
    scenes: List[Dict] = []

    for i in range(n_scenes):
        base_xy = rng.rand(n_nodes, 2).astype(np.float32)

        frames = []
        for t in range(temporal_window):
            noise = rng.randn(n_nodes, 2).astype(np.float32) * 0.02
            positions = base_xy + noise * (t + 1)
            dx = noise[:, 0]
            dy = noise[:, 1]
            density = rng.rand(n_nodes).astype(np.float32) * 0.3

            features = np.zeros((n_nodes, n_features), dtype=np.float32)
            features[:, 0] = positions[:, 0]
            features[:, 1] = positions[:, 1]
            features[:, 2] = dx
            features[:, 3] = dy
            features[:, 4] = density

            # Build edges
            coords = positions
            src, dst = [], []
            for a in range(n_nodes):
                for b in range(a + 1, n_nodes):
                    if np.linalg.norm(coords[a] - coords[b]) < 0.5:
                        src.extend([a, b])
                        dst.extend([b, a])
            if not src:
                src, dst = [0, 1], [1, 0]

            edge_index = np.array([src, dst], dtype=np.int64)
            frames.append({"x": features, "edge_index": edge_index})

        # Decide label and inject anomaly signatures
        r = rng.rand()
        if r < (1.0 - anomaly_ratio):
            label = "NORMAL"
            target = frames[-1]["x"][:, :2] + rng.randn(n_nodes, 2).astype(np.float32) * 0.02
        elif r < (1.0 - anomaly_ratio / 2):
            label = "UNSTABLE"
            target = frames[-1]["x"][:, :2] + rng.randn(n_nodes, 2).astype(np.float32) * 0.15
            frames[-1]["x"][:, 4] = rng.rand(n_nodes).astype(np.float32) * 0.5 + 0.4
            frames[-1]["x"][:, 2:4] *= 3.0
        else:
            label = "STAMPEDE"
            target = frames[-1]["x"][:, :2] + rng.randn(n_nodes, 2).astype(np.float32) * 0.4
            frames[-1]["x"][:, 4] = rng.rand(n_nodes).astype(np.float32) * 0.3 + 0.7
            frames[-1]["x"][:, 2:4] = rng.randn(n_nodes, 2).astype(np.float32) * 0.5

        scenes.append({
            "frames": frames,
            "target": target.astype(np.float32),
            "label": label,
        })

    counts = {}
    for s in scenes:
        counts[s["label"]] = counts.get(s["label"], 0) + 1
    logger.info("Generated %d scenes: %s", n_scenes, counts)

    return scenes


# ============================================================================
# Main pipeline test
# ============================================================================
def run_pipeline_test(
    checkpoint_path: str,
    output_dir: str,
    n_samples: int = 200,
    device_str: str = "cpu",
) -> Optional[Dict]:
    """Run end-to-end pipeline test and return all metrics."""

    # --- Import all pipeline modules ---
    try:
        import torch
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            cohen_kappa_score,
            confusion_matrix,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            precision_score,
            recall_score,
            r2_score,
            roc_auc_score,
        )
    except ImportError as exc:
        logger.error("Missing dependency: %s", exc)
        return None

    try:
        from models.stgnn import STGNN
    except ImportError:
        logger.error("Cannot import STGNN. Ensure src/models/stgnn.py exists.")
        return None

    try:
        from temporal_buffer import TemporalGraphBuffer
    except ImportError:
        logger.error("Cannot import TemporalGraphBuffer.")
        return None

    try:
        from crowd_metrics import CrowdMetrics
    except ImportError:
        logger.error("Cannot import CrowdMetrics.")
        return None

    try:
        from alert_logic import StampedeAlert
    except ImportError:
        logger.error("Cannot import StampedeAlert.")
        return None

    # --- Load STGNN model ---
    if not os.path.isfile(checkpoint_path):
        logger.warning(
            "Checkpoint not found at %s — using random weights (demo only).",
            checkpoint_path,
        )
        model = STGNN(**STGNN_CONFIG)
    else:
        logger.info("Loading checkpoint from %s …", checkpoint_path)
        model = STGNN(**STGNN_CONFIG)
        try:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            elif isinstance(state, dict) and "state_dict" in state:
                model.load_state_dict(state["state_dict"])
            else:
                model.load_state_dict(state)
            logger.info("Checkpoint loaded.")
        except Exception as exc:
            logger.error("Failed to load checkpoint: %s", exc)

    device = torch.device(device_str)
    model = model.to(device)
    model.eval()

    # --- Generate scenes ---
    scenes = generate_crowd_scenes(n_scenes=n_samples)

    # --- Module tests ---
    print("\n" + "█" * 60)
    print("  MODULE INTEGRATION CHECKS")
    print("█" * 60)

    module_checks = []

    # 1. TemporalGraphBuffer check
    try:
        buf = TemporalGraphBuffer(window_size=TEMPORAL_WINDOW)
        test_scene = scenes[0]
        for frame in test_scene["frames"]:
            result = buf.push(frame)
        assert result is not None, "Buffer did not produce output"
        assert result.shape[0] == 1, f"Batch dim wrong: {result.shape}"
        assert result.shape[1] == TEMPORAL_WINDOW, f"Time dim wrong: {result.shape}"
        module_checks.append(("TemporalGraphBuffer", "✅ PASS"))
    except Exception as exc:
        module_checks.append(("TemporalGraphBuffer", f"❌ FAIL: {exc}"))

    # 2. CrowdMetrics check
    try:
        cm_result = CrowdMetrics.compute(test_scene["frames"][-1])
        assert "mean_density" in cm_result, "Missing mean_density"
        assert "motion_entropy" in cm_result, "Missing motion_entropy"
        assert "mean_speed" in cm_result, "Missing mean_speed"
        module_checks.append(("CrowdMetrics", "✅ PASS"))
    except Exception as exc:
        module_checks.append(("CrowdMetrics", f"❌ FAIL: {exc}"))

    # 3. StampedeAlert check
    try:
        alert = StampedeAlert()
        state = alert.update(0.01, cm_result)
        assert state in ("NORMAL", "UNSTABLE", "STAMPEDE"), f"Invalid state: {state}"
        module_checks.append(("StampedeAlert", "✅ PASS"))
    except Exception as exc:
        module_checks.append(("StampedeAlert", f"❌ FAIL: {exc}"))

    # 4. STGNN inference check
    try:
        with torch.no_grad():
            x_t = torch.tensor(result, dtype=torch.float32).to(device)
            e_t = torch.tensor(
                test_scene["frames"][-1]["edge_index"], dtype=torch.long
            ).to(device)
            out = model(x_t, e_t)
            assert out.shape[-1] == 2, f"Output features wrong: {out.shape}"
        module_checks.append(("STGNN Inference", "✅ PASS"))
    except Exception as exc:
        module_checks.append(("STGNN Inference", f"❌ FAIL: {exc}"))

    print(_print_table("MODULE STATUS", module_checks))

    failed = [m for m in module_checks if "FAIL" in m[1]]
    if failed:
        logger.error("%d module(s) failed. Fix before running full evaluation.", len(failed))
        return None

    # --- Full pipeline evaluation ---
    print("\n" + "█" * 60)
    print("  END-TO-END PIPELINE EVALUATION")
    print("█" * 60)

    all_true_pos: List[np.ndarray] = []
    all_pred_pos: List[np.ndarray] = []
    all_anomaly_scores: List[float] = []
    true_labels: List[str] = []
    pred_labels: List[str] = []
    pipeline_times: List[float] = []

    alert_engine = StampedeAlert()

    logger.info("Running full pipeline on %d scenes …", len(scenes))

    with torch.no_grad():
        for scene in scenes:
            t_start = time.perf_counter()

            # Step 1: Push frames through TemporalGraphBuffer
            buf = TemporalGraphBuffer(window_size=TEMPORAL_WINDOW)
            x_seq = None
            for frame in scene["frames"]:
                x_seq = buf.push(frame)

            if x_seq is None:
                continue

            # Step 2: STGNN inference
            edge_index = scene["frames"][-1]["edge_index"]
            x_t = torch.tensor(x_seq, dtype=torch.float32).to(device)
            e_t = torch.tensor(edge_index, dtype=torch.long).to(device)

            try:
                y_pred = model(x_t, e_t).squeeze(0).cpu().numpy()
            except Exception:
                continue

            y_true = scene["target"]

            # Step 3: Anomaly score
            anomaly_score = float(np.mean((y_pred - y_true) ** 2))

            # Step 4: CrowdMetrics
            crowd_met = CrowdMetrics.compute(scene["frames"][-1])

            # Step 5: StampedeAlert
            pred_state = alert_engine.update(anomaly_score, crowd_met)

            t_end = time.perf_counter()

            # Collect results
            all_true_pos.append(y_true)
            all_pred_pos.append(y_pred)
            all_anomaly_scores.append(anomaly_score)
            true_labels.append(scene["label"])
            pred_labels.append(pred_state)
            pipeline_times.append((t_end - t_start) * 1000)

    if not all_true_pos:
        logger.error("No valid predictions — aborting.")
        return None

    # --- Compute metrics ---
    all_true_np = np.concatenate(all_true_pos, axis=0)
    all_pred_np = np.concatenate(all_pred_pos, axis=0)

    # Regression
    mse = float(mean_squared_error(all_true_np, all_pred_np))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(all_true_np, all_pred_np))
    r2 = float(r2_score(all_true_np, all_pred_np))

    mask = np.abs(all_true_np) > 1e-8
    mape = float(np.mean(np.abs((all_true_np[mask] - all_pred_np[mask]) / all_true_np[mask])) * 100) if mask.any() else 0.0

    # Binary classification
    true_binary = [0 if l == "NORMAL" else 1 for l in true_labels]
    pred_binary = [0 if l == "NORMAL" else 1 for l in pred_labels]

    bin_accuracy = float(accuracy_score(true_binary, pred_binary))
    bin_precision = float(precision_score(true_binary, pred_binary, zero_division=0))
    bin_recall = float(recall_score(true_binary, pred_binary, zero_division=0))
    bin_f1 = float(f1_score(true_binary, pred_binary, zero_division=0))

    try:
        auc_roc = float(roc_auc_score(true_binary, all_anomaly_scores)) if len(set(true_binary)) > 1 else 0.0
    except Exception:
        auc_roc = 0.0

    # Multi-class classification
    class_names = ["NORMAL", "UNSTABLE", "STAMPEDE"]
    mc_accuracy = float(accuracy_score(true_labels, pred_labels))
    mc_precision = float(precision_score(true_labels, pred_labels, average="weighted", zero_division=0))
    mc_recall = float(recall_score(true_labels, pred_labels, average="weighted", zero_division=0))
    mc_f1 = float(f1_score(true_labels, pred_labels, average="weighted", zero_division=0))
    kappa = float(cohen_kappa_score(true_labels, pred_labels))

    per_class_f1 = f1_score(true_labels, pred_labels, labels=class_names, average=None, zero_division=0)
    per_class_f1_dict = {c: round(float(f), 4) for c, f in zip(class_names, per_class_f1)}

    cls_report = classification_report(
        true_labels, pred_labels,
        labels=class_names, target_names=class_names, zero_division=0,
    )
    cm = confusion_matrix(true_labels, pred_labels, labels=class_names)

    # Latency
    avg_latency = float(np.mean(pipeline_times))
    p95_latency = float(np.percentile(pipeline_times, 95))
    throughput = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    # --- Console output ---
    print()
    print(_print_table("TRAJECTORY REGRESSION", [
        ("MSE", f"{mse:.6f}"),
        ("RMSE", f"{rmse:.6f}"),
        ("MAE", f"{mae:.6f}"),
        ("R²", f"{r2:.6f}"),
        ("MAPE (%)", f"{mape:.4f}"),
    ]))
    print()
    print(_print_table("BINARY CLASSIFICATION (Normal vs Anomaly)", [
        ("Accuracy", f"{bin_accuracy:.4f}"),
        ("Precision", f"{bin_precision:.4f}"),
        ("Recall", f"{bin_recall:.4f}"),
        ("F1-Score", f"{bin_f1:.4f}"),
        ("AUC-ROC", f"{auc_roc:.4f}"),
    ]))
    print()
    print(_print_table("MULTI-CLASS CLASSIFICATION", [
        ("Accuracy", f"{mc_accuracy:.4f}"),
        ("Precision (weighted)", f"{mc_precision:.4f}"),
        ("Recall (weighted)", f"{mc_recall:.4f}"),
        ("F1-Score (weighted)", f"{mc_f1:.4f}"),
        ("Cohen's Kappa", f"{kappa:.4f}"),
    ]))
    print()
    print(_print_table("PER-CLASS F1 SCORES", [
        (c, f"{f:.4f}") for c, f in per_class_f1_dict.items()
    ]))
    print()
    print("CLASSIFICATION REPORT (per class):")
    print(cls_report)
    print("CONFUSION MATRIX:")
    header = "           " + "  ".join(f"{c:>10s}" for c in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = "  ".join(f"{cm[i, j]:>10d}" for j in range(len(class_names)))
        print(f"  {name:<10s} {row}")
    print()
    print(_print_table("PIPELINE PERFORMANCE", [
        ("Avg latency", f"{avg_latency:.3f} ms"),
        ("P95 latency", f"{p95_latency:.3f} ms"),
        ("Throughput", f"{throughput:.1f} samples/sec"),
        ("Total evaluated", str(len(all_true_pos))),
    ]))

    # --- Save results ---
    results = {
        "test_type": "end_to_end_pipeline",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": checkpoint_path,
        "num_samples": n_samples,
        "num_evaluated": len(all_true_pos),
        "regression": {
            "MSE": round(mse, 6), "RMSE": round(rmse, 6),
            "MAE": round(mae, 6), "R2": round(r2, 6),
            "MAPE_percent": round(mape, 4),
        },
        "binary_classification": {
            "Accuracy": round(bin_accuracy, 4),
            "Precision": round(bin_precision, 4),
            "Recall": round(bin_recall, 4),
            "F1_Score": round(bin_f1, 4),
            "AUC_ROC": round(auc_roc, 4),
        },
        "multiclass_classification": {
            "Accuracy": round(mc_accuracy, 4),
            "Precision_weighted": round(mc_precision, 4),
            "Recall_weighted": round(mc_recall, 4),
            "F1_Score_weighted": round(mc_f1, 4),
            "Cohens_Kappa": round(kappa, 4),
            "Per_class_F1": per_class_f1_dict,
        },
        "pipeline_performance": {
            "avg_latency_ms": round(avg_latency, 3),
            "p95_latency_ms": round(p95_latency, 3),
            "throughput_samples_per_sec": round(throughput, 1),
        },
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "module_checks": {m[0]: m[1] for m in module_checks},
    }

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"e2e_pipeline_metrics_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info("Results saved → %s", json_path)

    # Save confusion matrix plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title("E2E Pipeline — Confusion Matrix", fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        thresh = cm.max() / 2.0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=13,
                )
        plt.tight_layout()
        cm_path = os.path.join(output_dir, f"e2e_confusion_matrix_{ts}.png")
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)
        logger.info("Confusion matrix plot → %s", cm_path)
    except Exception as exc:
        logger.warning("Could not save confusion matrix plot: %s", exc)

    print(f"\n✅ End-to-end pipeline test complete. Results → {output_dir}")
    return results


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline test for STGNN crowd anomaly detection.",
    )
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT,
        help="Path to STGNN .pth checkpoint",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Directory to save results",
    )
    parser.add_argument(
        "--n-samples", type=int, default=200,
        help="Number of synthetic test scenes",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="Device for inference",
    )
    args = parser.parse_args()

    result = run_pipeline_test(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        device_str=args.device,
    )
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
