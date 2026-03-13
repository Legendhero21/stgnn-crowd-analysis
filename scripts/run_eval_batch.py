"""
Headless Evaluation Runner — Aggregate Metrics for Report Table
Runs the full STGNN crowd anomaly detection pipeline on a YOLO model + video combo,
or runs YOLO formal validation on the labeled dataset.

Usage:
  # Full pipeline run
  python scripts/run_eval_batch.py --yolo-model models/yolo11n_person_best.pt --video data/videos/test_crowd.mp4 --tag yolo11n_yt

  # YOLO-only validation on labeled dataset
  python scripts/run_eval_batch.py --yolo-val-only --yolo-model models/yolo11n_person_best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — ensure src/ is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import onnxruntime as ort
from scipy.spatial.distance import cdist

from alert_logic import StampedeAlert
from crowd_metrics import CrowdMetrics
from temporal_buffer import TemporalGraphBuffer
from yolo_detector import YOLODetector

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - EVAL - %(levelname)s - %(message)s",
)
logger = logging.getLogger("EVAL")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_STGNN_ONNX = os.path.join(PROJECT_ROOT, "outputs", "evaluation", "stgnn_final.onnx")
DEFAULT_YOLO_DATA_YAML = os.path.join(PROJECT_ROOT, "data", "yolo_from_mat", "data.yaml")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "eval_results")

GRAPH_RADIUS = 0.05
MIN_NODES = 2
TEMPORAL_WINDOW = 5
ANOMALY_THRESHOLD_WARNING = 0.05
ANOMALY_THRESHOLD_CRITICAL = 0.15


# ============================================================
# Graph Builder (same as src/run_pipeline_realtime.py)
# ============================================================
class RealtimeGraphBuilder:
    def __init__(self, radius: float) -> None:
        self.radius = float(radius)
        self.prev_coords: Optional[np.ndarray] = None

    def build_graph(
        self,
        detections: List[Tuple[float, float]],
        frame_shape: Tuple[int, int],
    ) -> Optional[Dict[str, np.ndarray]]:
        if not detections or len(detections) < MIN_NODES:
            self.prev_coords = None
            return None

        h, w = frame_shape
        if h <= 0 or w <= 0:
            self.prev_coords = None
            return None

        coords = np.asarray(detections, dtype=np.float32).copy()
        if coords.ndim != 2 or coords.shape[1] != 2:
            self.prev_coords = None
            return None

        coords[:, 0] /= float(w)
        coords[:, 1] /= float(h)

        if self.prev_coords is not None and len(self.prev_coords) == len(coords):
            velocity = coords - self.prev_coords
        else:
            velocity = np.zeros_like(coords, dtype=np.float32)

        edge_index = self._build_edges(coords)
        density = self._compute_density(len(coords), edge_index)

        features = np.hstack([coords, velocity, density[:, None]]).astype(np.float32)
        self.prev_coords = coords.copy()

        return {"x": features, "edge_index": edge_index}

    def _build_edges(self, coords: np.ndarray) -> np.ndarray:
        if coords.size == 0:
            return np.zeros((2, 0), dtype=np.int64)
        dists = cdist(coords, coords)
        row, col = np.where((dists < self.radius) & (dists > 0.0))
        if row.size == 0:
            return np.zeros((2, 0), dtype=np.int64)
        return np.stack([row, col], axis=0).astype(np.int64)

    def _compute_density(self, n: int, edge_index: np.ndarray) -> np.ndarray:
        density = np.zeros(n, dtype=np.float32)
        if edge_index.shape[1] > 0:
            uniq, cnt = np.unique(edge_index[0], return_counts=True)
            density[uniq] = cnt
        if n > 1:
            density /= float(n - 1)
        return density


# ============================================================
# STGNN Inference (same as src/run_pipeline_realtime.py)
# ============================================================
class STGNNInference:
    def __init__(self, model_path: str) -> None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

    def predict_from_sequence(self, x_seq: np.ndarray, edge_index: np.ndarray) -> float:
        if x_seq is None or x_seq.size == 0:
            return 0.0
        if edge_index is None or edge_index.size == 0:
            return 0.0
        if x_seq.ndim != 4 or x_seq.shape[0] != 1:
            return 0.0
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            return 0.0

        try:
            outputs = self.session.run(
                self.output_names,
                {
                    self.input_names[0]: x_seq.astype(np.float32, copy=False),
                    self.input_names[1]: edge_index.astype(np.int64, copy=False),
                },
            )
        except Exception:
            return 0.0

        preds = np.asarray(outputs[0], dtype=np.float32)
        if preds.ndim == 3 and preds.shape[0] == 1:
            preds = preds[0]
        elif preds.ndim != 2:
            try:
                preds = preds.reshape(preds.shape[-2], preds.shape[-1])
            except Exception:
                return 0.0

        last_x = x_seq[0, -1, :, :2]
        if preds.shape[0] != last_x.shape[0]:
            return 0.0

        mse = float(np.mean((preds[:, :2] - last_x) ** 2))
        return mse


# ============================================================
# YOLO Formal Validation
# ============================================================
def run_yolo_validation(model_path: str, data_yaml: str, conf: float = 0.4) -> dict:
    """Run Ultralytics YOLO val() and return aggregate metrics."""
    from ultralytics import YOLO

    logger.info("=" * 70)
    logger.info("YOLO FORMAL VALIDATION")
    logger.info("Model:   %s", model_path)
    logger.info("Dataset: %s", data_yaml)
    logger.info("=" * 70)

    model = YOLO(model_path)

    # Determine correct classes for detection
    # VisDrone class 0 = pedestrian, standard YOLO class 0 = person
    is_visdrone = "visdrone" in os.path.basename(model_path).lower()

    results = model.val(
        data=data_yaml,
        conf=conf,
        iou=0.5,
        device="cuda",
        verbose=True,
    )

    # Extract metrics
    metrics = {
        "model": os.path.basename(model_path),
        "model_type": "visdrone" if is_visdrone else "yolo11n",
        "precision": float(results.box.mp),       # mean precision
        "recall": float(results.box.mr),          # mean recall
        "f1": float(2 * results.box.mp * results.box.mr / (results.box.mp + results.box.mr + 1e-8)),
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
    }

    logger.info("")
    logger.info("=" * 70)
    logger.info("YOLO VALIDATION RESULTS")
    logger.info("=" * 70)
    logger.info("Model:       %s", metrics["model"])
    logger.info("Precision:   %.4f", metrics["precision"])
    logger.info("Recall:      %.4f", metrics["recall"])
    logger.info("F1:          %.4f", metrics["f1"])
    logger.info("mAP@50:      %.4f", metrics["mAP50"])
    logger.info("mAP@50-95:   %.4f", metrics["mAP50_95"])
    logger.info("=" * 70)

    return metrics


# ============================================================
# Full Pipeline Run (headless)
# ============================================================
def run_pipeline(
    yolo_model_path: str,
    video_path: str,
    stgnn_model_path: str,
    tag: str,
    output_dir: str,
    federated: bool = False,
) -> dict:
    """
    Run the complete pipeline headlessly and return aggregate metrics.
    """
    logger.info("=" * 70)
    logger.info("FULL PIPELINE EVALUATION")
    logger.info("YOLO:    %s", yolo_model_path)
    logger.info("Video:   %s", video_path)
    logger.info("STGNN:   %s", stgnn_model_path)
    logger.info("Tag:     %s", tag)
    logger.info("Fed:     %s", "YES" if federated else "NO")
    logger.info("=" * 70)

    # Validate files
    for path, label in [
        (yolo_model_path, "YOLO model"),
        (video_path, "Video"),
        (stgnn_model_path, "STGNN ONNX"),
    ]:
        if not os.path.isfile(path):
            logger.error("%s not found: %s", label, path)
            sys.exit(1)

    # Initialize components
    device = "cuda"
    is_visdrone = "visdrone" in os.path.basename(yolo_model_path).lower()

    yolo = YOLODetector(model_path=yolo_model_path, conf_threshold=0.4, device=device)
    graph_builder = RealtimeGraphBuilder(GRAPH_RADIUS)
    stgnn = STGNNInference(stgnn_model_path)
    temporal_buffer = TemporalGraphBuffer(window_size=TEMPORAL_WINDOW)
    alert_engine = StampedeAlert()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Failed to open video: %s", video_path)
        sys.exit(1)

    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("Total frames in video: %d", total_frames_in_video)

    # Accumulators
    frame_idx = 0
    person_counts: List[int] = []
    anomaly_scores: List[float] = []
    states: List[str] = []
    densities: List[float] = []
    entropies: List[float] = []
    speeds: List[float] = []

    # Resource tracking
    import psutil
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            centers, _ = yolo.detect_persons_with_boxes(frame)
            centers = centers or []

            graph = graph_builder.build_graph(centers, frame.shape[:2])
            x_seq = temporal_buffer.push(graph)

            if x_seq is not None and graph is not None:
                anomaly = stgnn.predict_from_sequence(x_seq, graph["edge_index"])
            else:
                anomaly = 0.0

            metrics = CrowdMetrics.compute(graph)
            state = alert_engine.update(anomaly, metrics)

            # Accumulate
            person_counts.append(len(centers))
            anomaly_scores.append(anomaly)
            states.append(state)
            densities.append(metrics.get("mean_density", 0.0))
            entropies.append(metrics.get("motion_entropy", 0.0))
            speeds.append(metrics.get("mean_speed", 0.0))

            if frame_idx % 500 == 0:
                elapsed = time.time() - t_start
                fps = frame_idx / elapsed if elapsed > 0 else 0
                logger.info(
                    "  Frame %d/%d | persons=%d | anomaly=%.5f | state=%s | FPS=%.1f",
                    frame_idx, total_frames_in_video,
                    len(centers), anomaly, state, fps,
                )

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cap.release()

    t_end = time.time()
    total_time = t_end - t_start
    mem_after = process.memory_info().rss / 1024 / 1024

    # YOLO stats
    yolo_stats = yolo.get_stats()

    # Compute aggregate metrics
    n = len(person_counts)
    if n == 0:
        logger.error("No frames processed!")
        return {}

    state_counts = {s: states.count(s) for s in ["NORMAL", "UNSTABLE", "STAMPEDE"]}

    agg = {
        "tag": tag,
        "yolo_model": os.path.basename(yolo_model_path),
        "model_type": "visdrone" if is_visdrone else "yolo11n",
        "video": os.path.basename(video_path),
        "federated": federated,
        "total_frames": n,
        "total_time_s": round(total_time, 2),
        "avg_fps": round(n / total_time, 2) if total_time > 0 else 0,
        "yolo_avg_inference_ms": round(yolo_stats.get("avg_inference_ms", 0.0), 2),
        # Person count stats
        "avg_persons": round(float(np.mean(person_counts)), 2),
        "max_persons": int(np.max(person_counts)),
        "min_persons": int(np.min(person_counts)),
        "std_persons": round(float(np.std(person_counts)), 2),
        # Anomaly stats
        "avg_anomaly": round(float(np.mean(anomaly_scores)), 6),
        "max_anomaly": round(float(np.max(anomaly_scores)), 6),
        "std_anomaly": round(float(np.std(anomaly_scores)), 6),
        # State distribution (%)
        "pct_normal": round(100 * state_counts.get("NORMAL", 0) / n, 2),
        "pct_unstable": round(100 * state_counts.get("UNSTABLE", 0) / n, 2),
        "pct_stampede": round(100 * state_counts.get("STAMPEDE", 0) / n, 2),
        # Crowd metrics
        "avg_density": round(float(np.mean(densities)), 4),
        "avg_entropy": round(float(np.mean(entropies)), 4),
        "avg_speed": round(float(np.mean(speeds)), 4),
        # Resource usage
        "mem_before_mb": round(mem_before, 1),
        "mem_after_mb": round(mem_after, 1),
        "mem_used_mb": round(mem_after - mem_before, 1),
        "gpu_device": device,
    }

    # Save JSON
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{tag}.json")
    with open(json_path, "w") as f:
        json.dump(agg, f, indent=2)
    logger.info("Saved aggregate metrics to: %s", json_path)

    # Print summary
    print("\n")
    print("=" * 70)
    print(f"  PIPELINE RESULTS — {tag}")
    print("=" * 70)
    print(f"  YOLO Model:        {agg['yolo_model']}")
    print(f"  Video:             {agg['video']}")
    print(f"  Federated:         {'YES' if federated else 'NO'}")
    print(f"  Total Frames:      {agg['total_frames']}")
    print(f"  Total Time:        {agg['total_time_s']}s")
    print(f"  Avg FPS:           {agg['avg_fps']}")
    print(f"  YOLO Inference:    {agg['yolo_avg_inference_ms']}ms/frame")
    print("-" * 70)
    print(f"  Avg Persons:       {agg['avg_persons']} (max={agg['max_persons']})")
    print(f"  Avg Anomaly:       {agg['avg_anomaly']}")
    print(f"  Max Anomaly:       {agg['max_anomaly']}")
    print("-" * 70)
    print(f"  NORMAL:            {agg['pct_normal']}%")
    print(f"  UNSTABLE:          {agg['pct_unstable']}%")
    print(f"  STAMPEDE:          {agg['pct_stampede']}%")
    print("-" * 70)
    print(f"  Avg Density:       {agg['avg_density']}")
    print(f"  Avg Entropy:       {agg['avg_entropy']}")
    print(f"  Avg Speed:         {agg['avg_speed']}")
    print("-" * 70)
    print(f"  Memory Used:       {agg['mem_used_mb']} MB")
    print("=" * 70)
    print("")

    return agg


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Evaluation runner — aggregate metrics for report table"
    )
    parser.add_argument(
        "--yolo-model", type=str, required=True,
        help="Path to YOLO .pt model"
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to input video (required unless --yolo-val-only)"
    )
    parser.add_argument(
        "--stgnn-model", type=str, default=DEFAULT_STGNN_ONNX,
        help=f"Path to STGNN ONNX model (default: {DEFAULT_STGNN_ONNX})"
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Tag for this run (used in output filenames)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--yolo-val-only", action="store_true",
        help="Only run YOLO validation on labeled dataset (no video pipeline)"
    )
    parser.add_argument(
        "--yolo-data", type=str, default=DEFAULT_YOLO_DATA_YAML,
        help=f"Path to YOLO data.yaml for validation (default: {DEFAULT_YOLO_DATA_YAML})"
    )
    parser.add_argument(
        "--federated", action="store_true",
        help="Mark this run as federated (for comparison table, placeholder)"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.yolo_val_only:
        # YOLO formal validation mode
        metrics = run_yolo_validation(args.yolo_model, args.yolo_data)
        # Save
        model_name = os.path.splitext(os.path.basename(args.yolo_model))[0]
        out_path = os.path.join(args.output_dir, f"yolo_val_{model_name}.json")
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved YOLO validation results to: %s", out_path)
    else:
        # Full pipeline mode
        if args.video is None:
            parser.error("--video is required unless using --yolo-val-only")

        tag = args.tag
        if tag is None:
            model_base = os.path.splitext(os.path.basename(args.yolo_model))[0]
            video_base = os.path.splitext(os.path.basename(args.video))[0]
            tag = f"{model_base}_{video_base}"

        run_pipeline(
            yolo_model_path=args.yolo_model,
            video_path=args.video,
            stgnn_model_path=args.stgnn_model,
            tag=tag,
            output_dir=args.output_dir,
            federated=args.federated,
        )


if __name__ == "__main__":
    main()
