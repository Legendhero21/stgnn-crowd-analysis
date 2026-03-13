"""
Model Evaluation Script
========================
Evaluates YOLO (person detection) and STGNN (crowd anomaly detection)
with Accuracy, Precision, Recall, F1-Score and additional metrics.

Usage:
    python scripts/evaluate_models.py
    python scripts/evaluate_models.py --yolo-only
    python scripts/evaluate_models.py --stgnn-only
    python scripts/evaluate_models.py --stgnn-checkpoint path/to/model.pth
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
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
logger = logging.getLogger("ModelEvaluator")

# ---------------------------------------------------------------------------
# Default paths (override via CLI flags)
# ---------------------------------------------------------------------------
DEFAULT_STGNN_CHECKPOINT = str(
    PROJECT_ROOT / "outputs" / "evaluation" / "stgnn_final_frozen.pth"
)
DEFAULT_YOLO_MODEL = str(PROJECT_ROOT / "models" / "yolo11n_person_best.pt")
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "evaluation")

# STGNN model config (must match the trained checkpoint)
STGNN_CONFIG = {
    "in_channels": 5,
    "hidden_channels": 64,
    "out_channels": 2,
    "num_layers": 3,
    "dropout": 0.1,
    "kernel_size": 3,
}


# ============================================================================
# Helper: pretty table printer
# ============================================================================
def _print_table(title: str, rows: List[Tuple[str, str]]) -> str:
    """Return a nicely formatted ASCII table."""
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
# 1. YOLO EVALUATION
# ============================================================================
def evaluate_yolo(
    model_path: str,
    data_yaml: Optional[str] = None,
    conf: float = 0.4,
    iou: float = 0.5,
) -> Optional[Dict]:
    """
    Evaluate YOLO person detection model using Ultralytics val().

    Args:
        model_path: Path to the .pt YOLO model.
        data_yaml: Path to a YOLO-format data.yaml with val split.
                   If None, attempts to use the model's default dataset.
        conf: Confidence threshold.
        iou: IoU threshold.

    Returns:
        Dict with Precision, Recall, F1, mAP metrics, or None on failure.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.warning("ultralytics not installed — skipping YOLO evaluation.")
        return None

    if not os.path.isfile(model_path):
        logger.warning("YOLO model not found at %s — skipping.", model_path)
        return None

    logger.info("Loading YOLO model from %s …", model_path)
    model = YOLO(model_path)

    val_kwargs = dict(
        conf=conf,
        iou=iou,
        verbose=False,
    )
    if data_yaml and os.path.isfile(data_yaml):
        val_kwargs["data"] = data_yaml

    try:
        results = model.val(**val_kwargs)
    except Exception as exc:
        logger.error("YOLO validation failed: %s", exc)
        logger.info(
            "Tip: make sure you have a labeled validation dataset in YOLO "
            "format. Pass --yolo-data path/to/data.yaml to specify it."
        )
        return None

    # Extract metrics from the results object
    precision = float(results.results_dict.get("metrics/precision(B)", 0.0))
    recall = float(results.results_dict.get("metrics/recall(B)", 0.0))
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    map50 = float(results.results_dict.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(results.results_dict.get("metrics/mAP50-95(B)", 0.0))

    metrics = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
    }

    report = _print_table(
        "YOLO PERSON DETECTION — EVALUATION",
        [
            ("Precision", f"{metrics['precision']:.4f}"),
            ("Recall", f"{metrics['recall']:.4f}"),
            ("F1-Score", f"{metrics['f1_score']:.4f}"),
            ("mAP@50", f"{metrics['mAP50']:.4f}"),
            ("mAP@50-95", f"{metrics['mAP50_95']:.4f}"),
        ],
    )
    print("\n" + report + "\n")
    return metrics


# ============================================================================
# 2. STGNN EVALUATION
# ============================================================================
def _generate_synthetic_test_data(
    n_samples: int = 200,
    temporal_window: int = 5,
    n_nodes: int = 10,
    n_features: int = 5,
    anomaly_ratio: float = 0.25,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate synthetic graph sequences for STGNN evaluation.

    Normal sequences have smooth trajectories; anomalous ones inject
    sudden jumps and speed spikes to simulate stampede-like events.

    Returns:
        List of dicts, each with keys:
            x_seq   : np.ndarray [1, T, N, F]
            edge_index : np.ndarray [2, E]
            target  : np.ndarray [N, 2]   (ground-truth next-frame positions)
            label   : str  "NORMAL" | "UNSTABLE" | "STAMPEDE"
    """
    rng = np.random.RandomState(seed)
    samples: List[Dict] = []

    for i in range(n_samples):
        # Base positions: random grid
        base_xy = rng.rand(n_nodes, 2).astype(np.float32)

        # Build temporal sequence
        x_seq = np.zeros((1, temporal_window, n_nodes, n_features), dtype=np.float32)
        for t in range(temporal_window):
            noise = rng.randn(n_nodes, 2).astype(np.float32) * 0.02
            positions = base_xy + noise * (t + 1)
            dx = noise[:, 0]
            dy = noise[:, 1]
            density = rng.rand(n_nodes).astype(np.float32) * 0.3
            x_seq[0, t, :, 0] = positions[:, 0]
            x_seq[0, t, :, 1] = positions[:, 1]
            x_seq[0, t, :, 2] = dx
            x_seq[0, t, :, 3] = dy
            x_seq[0, t, :, 4] = density

        # Decide label
        r = rng.rand()
        if r < (1.0 - anomaly_ratio):
            label = "NORMAL"
            # Target is a smooth continuation
            target = x_seq[0, -1, :, :2] + rng.randn(n_nodes, 2).astype(np.float32) * 0.02
        elif r < (1.0 - anomaly_ratio / 2):
            label = "UNSTABLE"
            # Moderate perturbation
            target = x_seq[0, -1, :, :2] + rng.randn(n_nodes, 2).astype(np.float32) * 0.15
            # Increase density and speed in sequence
            x_seq[0, -1, :, 4] = rng.rand(n_nodes).astype(np.float32) * 0.5 + 0.4
            x_seq[0, -1, :, 2:4] *= 3.0
        else:
            label = "STAMPEDE"
            # Large perturbation
            target = x_seq[0, -1, :, :2] + rng.randn(n_nodes, 2).astype(np.float32) * 0.4
            # Very high density and chaotic motion
            x_seq[0, -1, :, 4] = rng.rand(n_nodes).astype(np.float32) * 0.3 + 0.7
            x_seq[0, -1, :, 2:4] = rng.randn(n_nodes, 2).astype(np.float32) * 0.5

        # Build simple radius-based edges
        coords = x_seq[0, -1, :, :2]
        edges_src, edges_dst = [], []
        for a in range(n_nodes):
            for b in range(a + 1, n_nodes):
                dist = np.linalg.norm(coords[a] - coords[b])
                if dist < 0.5:
                    edges_src.extend([a, b])
                    edges_dst.extend([b, a])
        if len(edges_src) == 0:
            # fallback: connect nearest pair
            edges_src, edges_dst = [0, 1], [1, 0]

        edge_index = np.array([edges_src, edges_dst], dtype=np.int64)

        samples.append({
            "x_seq": x_seq,
            "edge_index": edge_index,
            "target": target.astype(np.float32),
            "label": label,
        })

    label_counts = {}
    for s in samples:
        label_counts[s["label"]] = label_counts.get(s["label"], 0) + 1
    logger.info("Synthetic dataset: %s", label_counts)

    return samples


def _classify_anomaly_score(
    anomaly_score: float,
    crowd_metrics: Dict[str, float],
    density_warn: float = 0.45,
    density_crit: float = 0.65,
    entropy_warn: float = 0.60,
    entropy_crit: float = 0.85,
    anomaly_crit: float = 0.08,
) -> str:
    """
    Replicate the StampedeAlert logic for a single frame (stateless version).

    Uses the same thresholds as alert_logic.StampedeAlert.
    """
    mean_density = float(crowd_metrics.get("mean_density", 0.0))
    motion_entropy = float(crowd_metrics.get("motion_entropy", 0.0))

    # STAMPEDE
    if (
        mean_density > density_crit
        and motion_entropy > entropy_crit
        and anomaly_score > anomaly_crit
    ):
        return "STAMPEDE"

    # UNSTABLE
    signals = 0
    if mean_density > density_warn:
        signals += 1
    if motion_entropy > entropy_warn:
        signals += 1
    if anomaly_score > anomaly_crit:
        signals += 1
    if signals >= 2:
        return "UNSTABLE"

    return "NORMAL"


def _compute_crowd_metrics_from_features(node_features: np.ndarray) -> Dict[str, float]:
    """
    Lightweight crowd metrics from node features [N, F].
    Mirrors CrowdMetrics.compute() but avoids import dependency issues.
    """
    if node_features.ndim != 2 or node_features.shape[0] == 0:
        return {"mean_density": 0.0, "motion_entropy": 0.0}

    dx = node_features[:, 2]
    dy = node_features[:, 3]
    density = node_features[:, 4]

    # Motion entropy
    angles = np.arctan2(dy, dx)
    hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi), density=True)
    hist = hist + 1e-6
    entropy = float(-np.sum(hist * np.log(hist)))

    return {
        "mean_speed": float(np.mean(np.sqrt(dx**2 + dy**2))),
        "mean_density": float(np.mean(density)),
        "max_density": float(np.max(density)),
        "motion_entropy": entropy,
    }


def evaluate_stgnn(
    checkpoint_path: str,
    output_dir: str,
    n_test_samples: int = 200,
    device_str: str = "cpu",
) -> Optional[Dict]:
    """
    Evaluate the STGNN model on test data with classification metrics.

    Metrics computed:
        - Regression: MSE, RMSE, MAE, R²
        - Classification (binary — Normal vs Anomaly):
            Accuracy, Precision, Recall, F1-Score
        - Classification (multi-class — NORMAL / UNSTABLE / STAMPEDE):
            Per-class Precision, Recall, F1-Score, Support
        - Confusion matrix

    Returns:
        Dict with all metrics, or None on failure.
    """
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

    # Import STGNN model
    try:
        from models.stgnn import STGNN
    except ImportError:
        logger.error("Cannot import STGNN model. Ensure src/models/stgnn.py exists.")
        return None

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    if not os.path.isfile(checkpoint_path):
        logger.warning(
            "STGNN checkpoint not found at %s — using randomly initialised model "
            "for demonstration. Metrics will NOT reflect real performance.",
            checkpoint_path,
        )
        model = STGNN(**STGNN_CONFIG)
    else:
        logger.info("Loading STGNN checkpoint from %s …", checkpoint_path)
        model = STGNN(**STGNN_CONFIG)
        try:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            elif isinstance(state, dict) and "state_dict" in state:
                model.load_state_dict(state["state_dict"])
            else:
                model.load_state_dict(state)
            logger.info("Checkpoint loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load checkpoint: %s — using random weights.", exc)

    device = torch.device(device_str)
    model = model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # Generate / load test data
    # ------------------------------------------------------------------
    test_data = _generate_synthetic_test_data(n_samples=n_test_samples)

    # ------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------
    all_true_positions: List[np.ndarray] = []
    all_pred_positions: List[np.ndarray] = []
    all_anomaly_scores: List[float] = []
    true_labels: List[str] = []
    pred_labels: List[str] = []
    inference_times: List[float] = []

    logger.info("Running STGNN inference on %d samples …", len(test_data))

    import time as _time

    with torch.no_grad():
        for sample in test_data:
            x_seq_t = torch.tensor(sample["x_seq"], dtype=torch.float32).to(device)
            edge_t = torch.tensor(sample["edge_index"], dtype=torch.long).to(device)

            try:
                t0 = _time.perf_counter()
                y_pred = model(x_seq_t, edge_t).squeeze(0).cpu().numpy()  # [N, 2]
                t1 = _time.perf_counter()
                inference_times.append((t1 - t0) * 1000)  # ms
            except Exception as exc:
                logger.debug("Inference error: %s — skipping sample.", exc)
                continue

            y_true = sample["target"]  # [N, 2]

            # Regression
            all_true_positions.append(y_true)
            all_pred_positions.append(y_pred)

            # Anomaly score = per-frame MSE
            anomaly_score = float(np.mean((y_pred - y_true) ** 2))
            all_anomaly_scores.append(anomaly_score)

            # Crowd metrics from last timestep features
            last_features = sample["x_seq"][0, -1, :, :]  # [N, F]
            crowd_met = _compute_crowd_metrics_from_features(last_features)

            # Classify using alert logic
            pred_label = _classify_anomaly_score(anomaly_score, crowd_met)
            pred_labels.append(pred_label)
            true_labels.append(sample["label"])

    if len(all_true_positions) == 0:
        logger.error("No valid predictions — evaluation aborted.")
        return None

    # ------------------------------------------------------------------
    # Regression metrics
    # ------------------------------------------------------------------
    all_true_np = np.concatenate(all_true_positions, axis=0)
    all_pred_np = np.concatenate(all_pred_positions, axis=0)

    mse = float(mean_squared_error(all_true_np, all_pred_np))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(all_true_np, all_pred_np))
    r2 = float(r2_score(all_true_np, all_pred_np))

    # MAPE (guard against zero true values)
    mask = np.abs(all_true_np) > 1e-8
    if mask.any():
        mape = float(np.mean(np.abs((all_true_np[mask] - all_pred_np[mask]) / all_true_np[mask])) * 100)
    else:
        mape = 0.0

    # Inference latency stats
    avg_latency = float(np.mean(inference_times)) if inference_times else 0.0
    p95_latency = float(np.percentile(inference_times, 95)) if inference_times else 0.0

    # ------------------------------------------------------------------
    # Binary classification: Normal (0) vs Anomaly (1)
    # ------------------------------------------------------------------
    true_binary = [0 if l == "NORMAL" else 1 for l in true_labels]
    pred_binary = [0 if l == "NORMAL" else 1 for l in pred_labels]

    bin_accuracy = float(accuracy_score(true_binary, pred_binary))
    bin_precision = float(precision_score(true_binary, pred_binary, zero_division=0))
    bin_recall = float(recall_score(true_binary, pred_binary, zero_division=0))
    bin_f1 = float(f1_score(true_binary, pred_binary, zero_division=0))

    # AUC-ROC (using anomaly scores as continuous predictions)
    try:
        if len(set(true_binary)) > 1:
            auc_roc = float(roc_auc_score(true_binary, all_anomaly_scores))
        else:
            auc_roc = 0.0
            logger.warning("Only one class present — AUC-ROC set to 0.0.")
    except Exception:
        auc_roc = 0.0

    # ------------------------------------------------------------------
    # Multi-class classification: NORMAL / UNSTABLE / STAMPEDE
    # ------------------------------------------------------------------
    class_names = ["NORMAL", "UNSTABLE", "STAMPEDE"]
    mc_accuracy = float(accuracy_score(true_labels, pred_labels))
    mc_precision = float(
        precision_score(true_labels, pred_labels, average="weighted", zero_division=0)
    )
    mc_recall = float(
        recall_score(true_labels, pred_labels, average="weighted", zero_division=0)
    )
    mc_f1 = float(
        f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
    )

    # Cohen's Kappa
    kappa = float(cohen_kappa_score(true_labels, pred_labels))

    # Per-class F1 scores
    per_class_f1 = f1_score(
        true_labels, pred_labels,
        labels=class_names, average=None, zero_division=0,
    )
    per_class_f1_dict = {c: round(float(f), 4) for c, f in zip(class_names, per_class_f1)}

    cls_report_str = classification_report(
        true_labels,
        pred_labels,
        labels=class_names,
        target_names=class_names,
        zero_division=0,
    )

    cm = confusion_matrix(true_labels, pred_labels, labels=class_names)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    results: Dict = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": checkpoint_path,
        "num_samples": len(test_data),
        "num_evaluated": len(all_true_positions),
        "regression": {
            "MSE": round(mse, 6),
            "RMSE": round(rmse, 6),
            "MAE": round(mae, 6),
            "R2": round(r2, 6),
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
        "inference_performance": {
            "avg_latency_ms": round(avg_latency, 3),
            "p95_latency_ms": round(p95_latency, 3),
            "total_samples": len(inference_times),
        },
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }

    # ---- Console report ----
    print("\n")
    print(
        _print_table(
            "STGNN — TRAJECTORY REGRESSION",
            [
                ("MSE", f"{mse:.6f}"),
                ("RMSE", f"{rmse:.6f}"),
                ("MAE", f"{mae:.6f}"),
                ("R²", f"{r2:.6f}"),
                ("MAPE (%)", f"{mape:.4f}"),
            ],
        )
    )
    print()
    print(
        _print_table(
            "STGNN — BINARY (Normal vs Anomaly)",
            [
                ("Accuracy", f"{bin_accuracy:.4f}"),
                ("Precision", f"{bin_precision:.4f}"),
                ("Recall", f"{bin_recall:.4f}"),
                ("F1-Score", f"{bin_f1:.4f}"),
                ("AUC-ROC", f"{auc_roc:.4f}"),
            ],
        )
    )
    print()
    print(
        _print_table(
            "STGNN — MULTI-CLASS (Normal / Unstable / Stampede)",
            [
                ("Accuracy", f"{mc_accuracy:.4f}"),
                ("Precision (weighted)", f"{mc_precision:.4f}"),
                ("Recall (weighted)", f"{mc_recall:.4f}"),
                ("F1-Score (weighted)", f"{mc_f1:.4f}"),
                ("Cohen's Kappa", f"{kappa:.4f}"),
            ],
        )
    )
    print()
    print(
        _print_table(
            "STGNN — PER-CLASS F1 SCORES",
            [(c, f"{f:.4f}") for c, f in per_class_f1_dict.items()],
        )
    )
    print()
    print(
        _print_table(
            "STGNN — INFERENCE PERFORMANCE",
            [
                ("Avg latency", f"{avg_latency:.3f} ms"),
                ("P95 latency", f"{p95_latency:.3f} ms"),
                ("Total samples", str(len(inference_times))),
            ],
        )
    )
    print()
    print("CLASSIFICATION REPORT (per class):")
    print(cls_report_str)

    print("CONFUSION MATRIX:")
    header = "           " + "  ".join(f"{c:>10s}" for c in class_names)
    print(header)
    for i, row_label in enumerate(class_names):
        row_vals = "  ".join(f"{cm[i, j]:>10d}" for j in range(len(class_names)))
        print(f"  {row_label:<10s} {row_vals}")
    print()

    # ---- Save confusion matrix plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title("STGNN Confusion Matrix", fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        # Annotate cells
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
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)
        logger.info("Confusion matrix saved to %s", cm_path)
    except Exception as exc:
        logger.warning("Could not save confusion matrix plot: %s", exc)

    # ---- Save JSON ----
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"evaluation_metrics_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info("Metrics saved to %s", json_path)

    # ---- Save text report ----
    report_path = os.path.join(output_dir, f"evaluation_report_{ts}.txt")
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  STGNN MODEL EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"  Timestamp : {results['timestamp']}\n")
        f.write(f"  Checkpoint: {checkpoint_path}\n")
        f.write(f"  Samples   : {results['num_evaluated']}/{results['num_samples']}\n\n")

        f.write("TRAJECTORY REGRESSION\n")
        f.write("-" * 60 + "\n")
        for k, v in results["regression"].items():
            f.write(f"  {k:<8s}: {v}\n")

        f.write("\nBINARY CLASSIFICATION (Normal vs Anomaly)\n")
        f.write("-" * 60 + "\n")
        for k, v in results["binary_classification"].items():
            f.write(f"  {k:<12s}: {v}\n")

        f.write("\nMULTI-CLASS CLASSIFICATION (Normal / Unstable / Stampede)\n")
        f.write("-" * 60 + "\n")
        for k, v in results["multiclass_classification"].items():
            if isinstance(v, dict):
                f.write(f"  {k:<20s}:\n")
                for ck, cv in v.items():
                    f.write(f"    {ck:<16s}: {cv}\n")
            else:
                f.write(f"  {k:<20s}: {v}\n")

        f.write("\nINFERENCE PERFORMANCE\n")
        f.write("-" * 60 + "\n")
        for k, v in results["inference_performance"].items():
            f.write(f"  {k:<20s}: {v}\n")

        f.write("\nPER-CLASS REPORT\n")
        f.write("-" * 60 + "\n")
        f.write(cls_report_str)
        f.write("\n")

    logger.info("Text report saved to %s", report_path)

    return results


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO + STGNN models with classification metrics."
    )
    parser.add_argument(
        "--yolo-model",
        default=DEFAULT_YOLO_MODEL,
        help="Path to YOLO .pt model",
    )
    parser.add_argument(
        "--yolo-data",
        default=None,
        help="Path to YOLO data.yaml with val split",
    )
    parser.add_argument(
        "--stgnn-checkpoint",
        default=DEFAULT_STGNN_CHECKPOINT,
        help="Path to STGNN .pth checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save evaluation outputs",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Number of synthetic test samples for STGNN",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for STGNN inference",
    )
    parser.add_argument(
        "--yolo-only",
        action="store_true",
        help="Run only YOLO evaluation",
    )
    parser.add_argument(
        "--stgnn-only",
        action="store_true",
        help="Run only STGNN evaluation",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_results: Dict = {}

    # ---- YOLO ----
    if not args.stgnn_only:
        print("\n" + "█" * 60)
        print("  YOLO PERSON DETECTION EVALUATION")
        print("█" * 60)
        yolo_metrics = evaluate_yolo(
            model_path=args.yolo_model,
            data_yaml=args.yolo_data,
        )
        if yolo_metrics:
            all_results["yolo"] = yolo_metrics

    # ---- STGNN ----
    if not args.yolo_only:
        print("\n" + "█" * 60)
        print("  STGNN ANOMALY DETECTION EVALUATION")
        print("█" * 60)
        stgnn_metrics = evaluate_stgnn(
            checkpoint_path=args.stgnn_checkpoint,
            output_dir=args.output_dir,
            n_test_samples=args.n_samples,
            device_str=args.device,
        )
        if stgnn_metrics:
            all_results["stgnn"] = stgnn_metrics

    if not all_results:
        logger.warning("No evaluations were completed successfully.")
        return 1

    print("\n✅ Evaluation complete. Results saved to:", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
