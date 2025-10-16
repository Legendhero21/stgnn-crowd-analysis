"""
YOLOv8s (VisDrone) ONNX Detector
Optimized for aerial/drone person detection with proper YOLOv8 output parsing.
"""

import cv2
import numpy as np
import onnxruntime as ort
import time
import logging
from typing import List, Tuple, Optional


# ==========================================================
# LOGGING CONFIG
# ==========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - YOLOv8-Detector - %(levelname)s - %(message)s"
)
logger = logging.getLogger("YOLOv8-Detector")


# ==========================================================
# YOLO DETECTOR CLASS
# ==========================================================
class YOLODetector:
    def __init__(
        self,
        model_path: str = "visdrone-yolov8s.onnx",
        conf_threshold: float = 0.10,  # Lower for aerial detection
        iou_threshold: float = 0.45,
        input_size: Tuple[int, int] = (640, 640),
        enable_gpu: bool = True,
    ):
        """
        Initialize YOLOv8 ONNX Runtime detector for VisDrone.

        Args:
            model_path: Path to ONNX file
            conf_threshold: Minimum confidence for detections (0.10 for aerial)
            iou_threshold: IoU threshold for NMS
            input_size: Resize dimensions (640, 640)
            enable_gpu: Use GPU if available
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size

        self.providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if enable_gpu
            else ["CPUExecutionProvider"]
        )

        # Initialize ONNX Runtime
        self._load_model()

        # Inference stats
        self.frame_count = 0
        self.inference_times = []

    # ==========================================================
    # MODEL LOADING
    # ==========================================================
    def _load_model(self):
        """Load ONNX model with optimizations."""
        logger.info(f"Loading YOLOv8 ONNX model from: {self.model_path}")
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        try:
            self.session = ort.InferenceSession(
                self.model_path, sess_options, providers=self.providers
            )
            
            # Get model info
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            # Log output shape for debugging
            output_shape = self.session.get_outputs()[0].shape
            logger.info(f"[OK] Model loaded successfully")
            logger.info(f"Providers: {self.session.get_providers()}")
            logger.info(f"Output shape: {output_shape}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")

    # ==========================================================
    # PREPROCESSING
    # ==========================================================
    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, float, float, int, int]:
        """
        Preprocess frame for ONNX inference with letterbox resize.
        
        Returns:
            input_tensor: (1, 3, H, W) normalized tensor
            r: scale ratio
            dw: width padding
            dh: height padding
            original_w: original frame width
            original_h: original frame height
        """
        original_h, original_w = frame.shape[:2]

        # Letterbox resize
        img, r, dw, dh = self._letterbox(frame, self.input_size)

        # Convert BGR → RGB, normalize to [0,1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)  # HWC → CHW
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

        return img, r, dw, dh, original_w, original_h

    def _letterbox(
        self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640), color: Tuple[int, int, int] = (114, 114, 114)
    ) -> Tuple[np.ndarray, float, float, float]:
        """
        Letterbox resize maintaining aspect ratio.
        
        Returns:
            resized image, scale ratio, dw padding, dh padding
        """
        shape = img.shape[:2]  # current [h, w]
        
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # Compute padding
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        # Resize
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        # Add border
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        return img, r, dw, dh

    # ==========================================================
    # INFERENCE + POSTPROCESS
    # ==========================================================
    def detect_persons_with_boxes(self, frame: np.ndarray) -> Tuple[List[Tuple[float, float]], np.ndarray]:
        """
        Detect persons (pedestrian + people) and return centers + boxes.
        
        Returns:
            centers: List of (x_center, y_center) tuples
            detections: Array of [x1, y1, x2, y2, conf, cls]
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty frame received")
            return [], np.array([])

        start = time.time()

        # Preprocess
        img, r, dw, dh, orig_w, orig_h = self._preprocess(frame)

        # Inference
        try:
            outputs = self.session.run(self.output_names, {self.input_name: img})[0]
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return [], np.array([])

        # Postprocess YOLOv8 output
        detections = self._postprocess_yolov8(outputs, r, dw, dh, orig_w, orig_h)
        
        # Extract centers
        centers = []
        if len(detections) > 0:
            centers = [(float((x1 + x2) / 2), float((y1 + y2) / 2)) 
                      for x1, y1, x2, y2, conf, cls in detections]

        # Timing stats
        inference_time = (time.time() - start) * 1000
        self.inference_times.append(inference_time)
        self.frame_count += 1
        
        if self.frame_count % 100 == 0:
            avg_ms = np.mean(self.inference_times[-100:])
            logger.info(f"Avg inference time (last 100 frames): {avg_ms:.2f}ms, FPS: {1000/avg_ms:.1f}")

        return centers, detections

    def _postprocess_yolov8(
        self, 
        output: np.ndarray, 
        r: float, 
        dw: float, 
        dh: float, 
        orig_w: int, 
        orig_h: int
    ) -> np.ndarray:
        """
        Postprocess YOLOv8 ONNX output.
        
        YOLOv8 output shape: (1, 84, 8400) for 80 COCO classes
        VisDrone: (1, 14, 8400) for 10 classes
        First 4 rows: [x_center, y_center, width, height]
        Remaining rows: class scores
        
        Returns:
            Array of [x1, y1, x2, y2, conf, cls]
        """
        # Remove batch dimension and transpose
        output = output[0]  # (num_classes+4, 8400)
        output = output.T   # (8400, num_classes+4)
        
        # Extract boxes and class scores
        boxes = output[:, :4]          # (8400, 4) - [x, y, w, h]
        scores = output[:, 4:]         # (8400, num_classes)
        
        # Get max class score and class ID for each detection
        class_ids = np.argmax(scores, axis=1)     # (8400,)
        confidences = np.max(scores, axis=1)      # (8400,)
        
        # Filter by confidence threshold
        mask = confidences > self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0:
            return np.array([])
        
        # Filter for VisDrone pedestrian (class 0) and people (class 1)
        person_mask = (class_ids == 0) | (class_ids == 1)
        boxes = boxes[person_mask]
        confidences = confidences[person_mask]
        class_ids = class_ids[person_mask]
        
        if len(boxes) == 0:
            return np.array([])
        
        # Convert from [x_center, y_center, w, h] to [x1, y1, x2, y2]
        x_center, y_center, width, height = boxes.T
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # Undo letterbox padding and scaling
        x1 = (x1 - dw) / r
        y1 = (y1 - dh) / r
        x2 = (x2 - dw) / r
        y2 = (y2 - dh) / r
        
        # Clip to original frame dimensions
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)
        
        # Stack into detections array
        detections = np.stack([x1, y1, x2, y2, confidences, class_ids], axis=1)
        
        # Apply NMS
        detections = self._nms(detections)
        
        return detections

    def _nms(self, detections: np.ndarray) -> np.ndarray:
        """Apply Non-Maximum Suppression."""
        if len(detections) == 0:
            return detections

        boxes = detections[:, :4]
        scores = detections[:, 4]
        
        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            # Compute IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            # Keep only boxes with IoU below threshold
            inds = np.where(iou <= self.iou_threshold)[0]
            order = order[inds + 1]

        return detections[keep]

    # ==========================================================
    # SIMPLE DETECTION METHOD (for compatibility)
    # ==========================================================
    def detect_persons(self, frame: np.ndarray) -> List[Tuple[float, float]]:
        """
        Simple method returning only center coordinates.
        For compatibility with existing code.
        """
        centers, _ = self.detect_persons_with_boxes(frame)
        return centers

    # ==========================================================
    # PERFORMANCE STATS
    # ==========================================================
    def get_stats(self) -> dict:
        """Get performance statistics."""
        if not self.inference_times:
            return {"frames_processed": 0, "avg_inference_time_ms": 0, "fps": 0}

        avg_ms = np.mean(self.inference_times)
        fps = 1000 / avg_ms if avg_ms > 0 else 0
        return {
            "frames_processed": self.frame_count,
            "avg_inference_time_ms": avg_ms,
            "fps": fps,
        }

    def reset_stats(self):
        """Reset performance counters."""
        self.inference_times.clear()
        self.frame_count = 0


# ==========================================================
# TEST (Optional)
# ==========================================================
if __name__ == "__main__":
    print("[INFO] Testing VisDrone YOLOv8s detector...")
    
    detector = YOLODetector(
        model_path="visdrone-yolov8s.onnx",
        conf_threshold=0.10,  # Low threshold for aerial detection
        iou_threshold=0.45,
        enable_gpu=True # Change to True if CUDA available
    )

    cap = cv2.VideoCapture(0)  # or video path
    print("[INFO] Press 'q' to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        centers, detections = detector.detect_persons_with_boxes(frame)

        # Draw bounding boxes
        for x1, y1, x2, y2, conf, cls in detections:
            label = "Pedestrian" if cls == 0 else "People"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw centers
        for x, y in centers:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

        # Display FPS
        stats = detector.get_stats()
        cv2.putText(frame, f"FPS: {stats['fps']:.1f} | People: {len(centers)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("VisDrone YOLOv8s Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Print final stats
    final_stats = detector.get_stats()
    print(f"\n[STATS] Frames processed: {final_stats['frames_processed']}")
    print(f"[STATS] Avg FPS: {final_stats['fps']:.2f}")
