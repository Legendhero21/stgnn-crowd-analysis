"""
YOLOv11n Detector (Ultralytics)
Clean integration for STGNN pipeline.
"""

import time
import logging
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# ==========================================================
# LOGGING
# ==========================================================
logger = logging.getLogger("YOLOv11-Detector")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - YOLOv11-Detector - %(levelname)s - %(message)s",
    )


# ==========================================================
# YOLO DETECTOR
# ==========================================================
class YOLODetector:
    def __init__(
        self,
        model_path: str = "models/yolo11n_person_best.pt",
        conf_threshold: float = 0.4,
        device: str = "cuda",  # "cuda" or "cpu"
    ):
        """
        YOLOv11n detector using Ultralytics runtime.

        Args:
            model_path: Path to trained YOLOv11n .pt model.
            conf_threshold: Confidence threshold for detections.
            device: 'cuda' or 'cpu'.
        """
        self.conf_threshold = float(conf_threshold)
        self.device = device

        # Load model
        try:
            self.model = YOLO(model_path)
        except Exception as exc:
            logger.error("Failed to load YOLO model from %s: %s", model_path, exc)
            raise

        try:
            self.model.to(device)
        except Exception as exc:
            logger.warning(
                "Failed to move model to device '%s' (%s). Falling back to CPU.",
                device,
                exc,
            )
            self.device = "cpu"
            self.model.to("cpu")

        # Performance stats
        self.frame_count = 0
        self.inference_times: List[float] = []

        logger.info("Loaded YOLOv11 model from %s", model_path)
        logger.info("Using device: %s", self.device)

    # ==========================================================
    # MAIN DETECTION API
    # ==========================================================
    def detect_persons_with_boxes(
        self, frame: np.ndarray
    ) -> Tuple[List[Tuple[float, float]], np.ndarray]:
        """
        Detect persons and return centers + boxes.

        Args:
            frame: BGR image (H, W, 3), np.uint8.

        Returns:
            centers: list of (x_center, y_center).
            detections: array [N, 5] with [x1, y1, x2, y2, conf].
        """
        if frame is None or frame.size == 0:
            logger.debug("Empty frame received; returning no detections.")
            return [], np.empty((0, 5), dtype=np.float32)

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Frame must be HxWx3 BGR image, got shape {frame.shape}")

        start = time.time()

        # YOLO inference
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
            classes=[0],  # only 'person' class
        )

        detections_list: List[List[float]] = []
        centers: List[Tuple[float, float]] = []

        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy
            confs = results[0].boxes.conf

            # Move to CPU and convert only once
            boxes = boxes.cpu().numpy()
            confs = confs.cpu().numpy()

            for (x1, y1, x2, y2), conf in zip(boxes, confs):
                detections_list.append([float(x1), float(y1), float(x2), float(y2), float(conf)])
                centers.append(
                    (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))
                )

        detections = (
            np.asarray(detections_list, dtype=np.float32)
            if detections_list
            else np.empty((0, 5), dtype=np.float32)
        )

        # Timing
        inference_time_ms = (time.time() - start) * 1000.0
        self.inference_times.append(inference_time_ms)
        self.frame_count += 1

        if self.frame_count % 100 == 0:
            avg_ms = float(np.mean(self.inference_times[-100:]))
            logger.info(
                "Avg inference over last 100 frames: %.2f ms | FPS: %.1f",
                avg_ms,
                1000.0 / avg_ms if avg_ms > 0 else 0.0,
            )

        return centers, detections

    # ==========================================================
    # COMPATIBILITY METHOD
    # ==========================================================
    def detect_persons(self, frame: np.ndarray) -> List[Tuple[float, float]]:
        """
        Compatibility wrapper for STGNN.
        Returns only person centers.
        """
        centers, _ = self.detect_persons_with_boxes(frame)
        return centers

    # ==========================================================
    # STATS
    # ==========================================================
    def get_stats(self) -> dict:
        """
        Return current performance stats.
        """
        if not self.inference_times:
            return {"frames": 0, "avg_inference_ms": 0.0, "fps": 0.0}

        avg_ms = float(np.mean(self.inference_times))
        return {
            "frames": self.frame_count,
            "avg_inference_ms": avg_ms,
            "fps": 1000.0 / avg_ms if avg_ms > 0 else 0.0,
        }

    def reset_stats(self) -> None:
        """
        Reset internal performance counters.
        """
        self.frame_count = 0
        self.inference_times.clear()
