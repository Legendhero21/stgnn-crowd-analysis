"""
Real-Time Crowd Anomaly Detection Pipeline
YOLOv11n (Ultralytics .pt) + Graph Builder + Temporal Buffer + STGNN (ONNX)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from scipy.spatial.distance import cdist

from alert_logic import StampedeAlert
from crowd_metrics import CrowdMetrics
from temporal_buffer import TemporalGraphBuffer

try:
    from yolo_detector import YOLODetector
except ImportError as exc:
    print(
        "[ERROR] Cannot import yolo_detector.py; "
        "check PYTHONPATH and file location."
    )
    print(f"Underlying error: {exc}")
    sys.exit(1)


# ==========================================================
# LOGGING
# ==========================================================
logger = logging.getLogger("PIPELINE")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - PIPELINE - %(levelname)s - %(message)s",
    )


# ==========================================================
# CONFIGURATION
# ==========================================================

YOLO_MODEL = os.getenv(
    "YOLO_MODEL_PATH",
    "D:/stgnn_project/models/yolo11n_person_best.pt",
)
STGNN_MODEL = os.getenv(
    "STGNN_MODEL_PATH",
    "D:/stgnn_project/outputs/evaluation/stgnn_final.onnx",
)
VIDEO_SOURCE = os.getenv(
    "VIDEO_SOURCE",
    "D:/stgnn_project/data/videos/mat_dataset_full.mp4",
)

OUTPUT_DIR = os.getenv(
    "PIPELINE_OUTPUT_DIR",
    "D:/stgnn_project/outputs/pipeline_results",
)

SAVE_OUTPUT_VIDEO = True
OUTPUT_FPS = 10

GRAPH_RADIUS = 0.05
MIN_NODES = 2

ANOMALY_THRESHOLD_WARNING = 0.05
ANOMALY_THRESHOLD_CRITICAL = 0.15

DISPLAY_GRAPH_EDGES = True
DISPLAY_FPS = True
FRAME_DELAY = 1  # ms for cv2.waitKey
TEMPORAL_WINDOW = 5


# ==========================================================
# VALIDATION
# ==========================================================


def validate_setup() -> None:
    logger.info("=" * 70)
    logger.info("REAL-TIME CROWD ANOMALY DETECTION PIPELINE")
    logger.info("=" * 70)

    missing = False
    if not os.path.isfile(YOLO_MODEL):
        logger.error("YOLOv11 model not found: %s", YOLO_MODEL)
        missing = True

    if not os.path.isfile(STGNN_MODEL):
        logger.error("STGNN model not found: %s", STGNN_MODEL)
        missing = True

    if not os.path.isfile(VIDEO_SOURCE):
        logger.error("Video source not found: %s", VIDEO_SOURCE)
        missing = True

    if missing:
        sys.exit(1)

    logger.info("[OK] YOLOv11 model: %s", YOLO_MODEL)
    logger.info("[OK] STGNN model: %s", STGNN_MODEL)
    logger.info("[OK] Video source: %s", VIDEO_SOURCE)
    logger.info("=" * 70 + "\n")


# ==========================================================
# GRAPH BUILDER
# ==========================================================


class RealtimeGraphBuilder:
    def __init__(self, radius: float) -> None:
        self.radius = float(radius)
        self.prev_coords: Optional[np.ndarray] = None

    def build_graph(
        self,
        detections: List[Tuple[float, float]],
        frame_shape: Tuple[int, int],
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Build a simple spatial graph from detection centers.

        Args:
            detections: list of (x, y) in pixel coordinates.
            frame_shape: (H, W).

        Returns:
            dict with 'x' (node features) and 'edge_index', or None if too few nodes.
        """
        if not detections or len(detections) < MIN_NODES:
            self.prev_coords = None
            return None

        h, w = frame_shape
        if h <= 0 or w <= 0:
            logger.warning("Invalid frame shape: %s", frame_shape)
            self.prev_coords = None
            return None

        coords = np.asarray(detections, dtype=np.float32).copy()
        if coords.ndim != 2 or coords.shape[1] != 2:
            logger.warning("Unexpected detections shape: %s", coords.shape)
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

        features = np.hstack(
            [
                coords,
                velocity,
                density[:, None],
            ]
        ).astype(np.float32)

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


# ==========================================================
# STGNN INFERENCE
# ==========================================================


class STGNNInference:
    def __init__(self, model_path: str) -> None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load STGNN ONNX model from %s: %s", model_path, exc)
            raise

        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

        if len(self.input_names) < 2:
            logger.warning(
                "STGNN model expects < 2 inputs; check ONNX signature. "
                "Inputs: %s",
                self.input_names,
            )

    def predict_from_sequence(
        self,
        x_seq: np.ndarray,
        edge_index: np.ndarray,
    ) -> float:
        """
        Args:
            x_seq: (1, T, N, F)
            edge_index: (2, E)

        Returns:
            anomaly_score: float (e.g., MSE between predicted and actual positions)
        """
        if x_seq is None or x_seq.size == 0:
            return 0.0
        if edge_index is None or edge_index.size == 0:
            return 0.0

        if x_seq.ndim != 4 or x_seq.shape[0] != 1:
            logger.warning("Unexpected x_seq shape: %s", x_seq.shape)
            return 0.0
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            logger.warning("Unexpected edge_index shape: %s", edge_index.shape)
            return 0.0

        try:
            outputs = self.session.run(
                self.output_names,
                {
                    self.input_names[0]: x_seq.astype(np.float32, copy=False),
                    self.input_names[1]: edge_index.astype(np.int64, copy=False),
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("STGNN inference failed: %s", exc)
            return 0.0

        preds = np.asarray(outputs[0], dtype=np.float32)

        # Expected shape: (1, N, F_pred) or (N, F_pred)
        if preds.ndim == 3 and preds.shape[0] == 1:
            preds = preds[0]
        elif preds.ndim != 2:
            logger.warning("Unexpected STGNN output shape: %s", preds.shape)
            try:
                preds = preds.reshape(preds.shape[-2], preds.shape[-1])
            except Exception:  # noqa: BLE001
                return 0.0

        last_x = x_seq[0, -1, :, :2]
        if preds.shape[0] != last_x.shape[0]:
            logger.warning(
                "Node count mismatch between preds (%d) and last_x (%d); returning 0.",
                preds.shape[0],
                last_x.shape[0],
            )
            return 0.0

        mse = float(np.mean((preds[:, :2] - last_x) ** 2))
        return mse


# ==========================================================
# VISUALIZATION
# ==========================================================


class CrowdVisualizer:
    COLORS = {
        "normal": (0, 255, 0),
        "warning": (0, 255, 255),
        "critical": (0, 0, 255),
    }

    def draw(
        self,
        frame: np.ndarray,
        centers: List[Tuple[float, float]],
        graph: Optional[Dict[str, np.ndarray]],
        anomaly: float,
        state: str,
    ) -> np.ndarray:
        vis = frame.copy()

        if anomaly > ANOMALY_THRESHOLD_CRITICAL:
            alert = "critical"
        elif anomaly > ANOMALY_THRESHOLD_WARNING:
            alert = "warning"
        else:
            alert = "normal"

        if DISPLAY_GRAPH_EDGES and graph is not None and len(centers) > 1:
            edge_index = graph.get("edge_index")
            if edge_index is not None and edge_index.size > 0:
                for s, d in edge_index.T:
                    if 0 <= s < len(centers) and 0 <= d < len(centers):
                        p1 = tuple(map(int, centers[s]))
                        p2 = tuple(map(int, centers[d]))
                        cv2.line(vis, p1, p2, (80, 80, 80), 1)

        for x, y in centers:
            cv2.circle(vis, (int(x), int(y)), 5, self.COLORS[alert], -1)

        cv2.putText(
            vis,
            f"People: {len(centers)} | Anomaly: {anomaly:.5f} | {alert.upper()}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self.COLORS[alert],
            2,
        )

        cv2.putText(
            vis,
            f"STATE: {state}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255)
            if state == "STAMPEDE"
            else (0, 255, 255)
            if state == "UNSTABLE"
            else (0, 255, 0),
            2,
        )

        return vis


# ==========================================================
# MAIN
# ==========================================================


def main() -> None:
    validate_setup()

    yolo = YOLODetector(
        model_path=YOLO_MODEL,
        conf_threshold=0.4,
        device="cuda",
    )

    graph_builder = RealtimeGraphBuilder(GRAPH_RADIUS)
    stgnn = STGNNInference(STGNN_MODEL)
    visualizer = CrowdVisualizer()
    temporal_buffer = TemporalGraphBuffer(window_size=TEMPORAL_WINDOW)
    alert_engine = StampedeAlert()

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        logger.error("Failed to open video: %s", VIDEO_SOURCE)
        return

    writer = None
    if SAVE_OUTPUT_VIDEO:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w <= 0 or h <= 0:
            logger.warning("Invalid video dimensions; disabling video writing.")
        else:
            out_path = os.path.join(OUTPUT_DIR, "pipeline_output.mp4")
            writer = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                OUTPUT_FPS,
                (w, h),
            )
            logger.info("Saving output video to: %s", out_path)

    frame_idx = 0
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video or read error.")
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

            vis = visualizer.draw(frame, centers, graph, anomaly, state)

            if DISPLAY_FPS:
                dt = time.time() - t0
                fps = frame_idx / dt if dt > 0 else 0.0
                cv2.putText(
                    vis,
                    f"FPS: {fps:.1f}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("Crowd Anomaly Detection", vis)

            if writer is not None:
                writer.write(vis)

            key = cv2.waitKey(FRAME_DELAY) & 0xFF
            if key == ord("q"):
                logger.info("User requested exit.")
                break

            if frame_idx % 200 == 0:
                logger.info(
                    "Frame %d | anomaly=%.5f | state=%s | mean_density=%.3f",
                    frame_idx,
                    anomaly,
                    state,
                    metrics.get("mean_density", 0.0),
                )

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

        stats = yolo.get_stats()
        logger.info("Frames processed: %d", stats.get("frames", frame_idx))
        logger.info("Avg FPS (YOLO): %.2f", stats.get("fps", 0.0))
        logger.info(
            "Avg YOLO inference: %.2f ms",
            stats.get("avg_inference_ms", 0.0),
        )
        logger.info("[SUCCESS] Pipeline finished.")


if __name__ == "__main__":
    main()
