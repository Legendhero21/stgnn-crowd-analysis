"""
Real-Time Crowd Anomaly Detection Pipeline
Integrates YOLOv8s (VisDrone) + Graph Builder + STGNN ONNX
Author: STGNN Project
Date: October 2025
"""

import os
import sys
import cv2
import numpy as np
import onnxruntime as ort
import time
import logging
from typing import List, Tuple, Optional, Dict

# Import YOLO detector
try:
    from yolo_detector import YOLODetector
except ImportError:
    print("[ERROR] Cannot import yolo_detector. Ensure yolo_detector.py is in the same directory.")
    sys.exit(1)

# Import scipy for graph building
try:
    from scipy.spatial.distance import cdist
except ImportError:
    print("[ERROR] scipy not installed. Run: pip install scipy")
    sys.exit(1)


# ==========================================================
# LOGGING CONFIGURATION
# ==========================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==========================================================
# CONFIGURATION
# ==========================================================

# Model paths
YOLO_MODEL = "D:/stgnn_project/src/visdrone-yolov8s.onnx"
STGNN_MODEL = "D:/stgnn_project/outputs/evaluation/stgnn_final.onnx"

# Video source (change as needed)
VIDEO_SOURCE = "D:/stgnn_project/data/videos/mat_dataset_full.mp4" # Dataset folder
# VIDEO_SOURCE = 0  # Uncomment for webcam
# VIDEO_SOURCE = "path/to/video.mp4"  # Uncomment for video file
# OUTPUT SETTINGS 
OUTPUT_DIR = "D:/stgnn_project/outputs/pipeline_results"
SAVE_OUTPUT_VIDEO = True  # Enable video saving
OUTPUT_FPS = 10  # Output video frame rate

# Graph building parameters
GRAPH_RADIUS = 0.05      # Spatial neighborhood radius (normalized coords)
MIN_NODES = 2            # Minimum nodes required to build graph

# Anomaly detection thresholds
ANOMALY_THRESHOLD_WARNING = 0.05   # Yellow alert
ANOMALY_THRESHOLD_CRITICAL = 0.15  # Red alert

# Display settings
DISPLAY_FPS = True
DISPLAY_GRAPH_EDGES = True
FRAME_DELAY = 100  # Milliseconds between frames (for dataset playback)


# ==========================================================
# SETUP VALIDATION
# ==========================================================
def validate_setup():
    """Validate all required files and paths exist."""
    logger.info("=" * 70)
    logger.info("REAL-TIME CROWD ANOMALY DETECTION PIPELINE")
    logger.info("=" * 70)
    
    errors = []
    
    # Check YOLO model
    if not os.path.exists(YOLO_MODEL):
        errors.append(f"YOLO model not found: {YOLO_MODEL}")
    else:
        size_mb = os.path.getsize(YOLO_MODEL) / (1024 * 1024)
        logger.info(f"[OK] YOLO Model: {YOLO_MODEL} ({size_mb:.2f} MB)")
    
    # Check STGNN model
    if not os.path.exists(STGNN_MODEL):
        errors.append(f"STGNN model not found: {STGNN_MODEL}")
    else:
        size_mb = os.path.getsize(STGNN_MODEL) / (1024 * 1024)
        logger.info(f"[OK] STGNN Model: {STGNN_MODEL} ({size_mb:.2f} MB)")
    
    # Check video source (if directory)
    if isinstance(VIDEO_SOURCE, str) and os.path.isdir(VIDEO_SOURCE):
        frame_files = [f for f in os.listdir(VIDEO_SOURCE) if f.endswith('.jpg')]
        if not frame_files:
            errors.append(f"No .jpg frames found in: {VIDEO_SOURCE}")
        else:
            logger.info(f"[OK] Dataset: {VIDEO_SOURCE} ({len(frame_files)} frames)")
    
    if errors:
        logger.error("\n[SETUP ERRORS]")
        for err in errors:
            logger.error(f"  - {err}")
        sys.exit(1)
    
    logger.info("=" * 70 + "\n")
    return True


# ==========================================================
# GRAPH BUILDER
# ==========================================================
class RealtimeGraphBuilder:
    """Build spatial-temporal graphs from detections."""
    
    def __init__(self, radius: float = GRAPH_RADIUS):
        """
        Initialize graph builder.
        
        Args:
            radius: Spatial neighborhood radius in normalized coordinates
        """
        self.radius = radius
        self.prev_coords = None
        
        logger.info(f"[INIT] Graph Builder (radius={radius})")
    
    def build_graph(
        self, 
        detections: List[Tuple[float, float]], 
        frame_shape: Tuple[int, int]
    ) -> Optional[Dict]:
        """
        Build graph from detections.
        
        Args:
            detections: List of (x, y) pixel coordinates
            frame_shape: (height, width) for normalization
        
        Returns:
            Dictionary with 'x' (node features) and 'edge_index' (edges)
            or None if insufficient nodes
        """
        if len(detections) < MIN_NODES:
            self.prev_coords = None
            return None
        
        frame_h, frame_w = frame_shape
        
        # Normalize coordinates to [0, 1]
        coords = np.array(detections, dtype=np.float32)
        coords[:, 0] /= frame_w
        coords[:, 1] /= frame_h
        
        # Compute velocity (displacement from previous frame)
        if self.prev_coords is not None and len(self.prev_coords) == len(coords):
            velocity = coords - self.prev_coords
        else:
            velocity = np.zeros_like(coords)
        
        # Build spatial edges
        edge_index = self._build_edges(coords)
        
        # Compute local density
        density = self._compute_density(len(coords), edge_index)
        
        # Construct feature matrix: [x, y, dx, dy, density]
        features = np.hstack([
            coords,           # Position (x, y)
            velocity,         # Velocity (dx, dy)
            density[:, None]  # Local density
        ]).astype(np.float32)
        
        # Store for next frame
        self.prev_coords = coords.copy()
        
        return {
            "x": features,
            "edge_index": edge_index
        }
    
    def _build_edges(self, coords: np.ndarray) -> np.ndarray:
        """Build edges based on spatial proximity."""
        if len(coords) < 2:
            return np.zeros((2, 0), dtype=np.int64)
        
        # Compute pairwise distances
        distances = cdist(coords, coords, metric='euclidean')
        
        # Find pairs within radius (exclude self-loops)
        row, col = np.where((distances < self.radius) & (distances > 0))
        
        if len(row) == 0:
            return np.zeros((2, 0), dtype=np.int64)
        
        return np.stack([row, col], axis=0).astype(np.int64)
    
    def _compute_density(self, num_nodes: int, edge_index: np.ndarray) -> np.ndarray:
        """Compute normalized local density (neighbor count)."""
        density = np.zeros(num_nodes, dtype=np.float32)
        
        if edge_index.shape[1] > 0:
            unique, counts = np.unique(edge_index[0], return_counts=True)
            density[unique] = counts
        
        # Normalize to [0, 1]
        max_neighbors = num_nodes - 1
        if max_neighbors > 0:
            density /= max_neighbors
        
        return density


# ==========================================================
# STGNN INFERENCE ENGINE
# ==========================================================
class STGNNInference:
    """STGNN model inference for anomaly detection."""
    
    def __init__(self, model_path: str):
        """
        Initialize STGNN inference engine.
        
        Args:
            model_path: Path to STGNN ONNX model
        """
        providers = ['CUDAExecutionProvider','CPUExecutionProvider']
        
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]
            
            logger.info(f"[INIT] STGNN Inference Engine")
            logger.info(f"       Inputs: {self.input_names}")
            
        except Exception as e:
            logger.error(f"Failed to load STGNN: {e}")
            raise
    
    def predict(self, graph: Optional[Dict]) -> Tuple[Optional[np.ndarray], float]:
        """
        Run STGNN inference and compute anomaly score.
        
        Args:
            graph: Dictionary with 'x' (features) and 'edge_index' (edges)
        
        Returns:
            predictions: Predicted positions (N, 2) or None
            anomaly_score: MSE-based anomaly score
        """
        if graph is None:
            return None, 0.0
        
        x = graph["x"]
        edge_index = graph["edge_index"]
        
        # Add batch and temporal dimensions: (1, 1, N, 5)
        x_input = np.expand_dims(np.expand_dims(x, 0), 0).astype(np.float32)
        edge_index_input = edge_index.astype(np.int64)
        
        try:
            # Run ONNX inference
            outputs = self.session.run(
                self.output_names,
                {
                    self.input_names[0]: x_input,
                    self.input_names[1]: edge_index_input
                }
            )
            
            # Extract predictions (remove batch/temporal dims)
            predictions = outputs[0].squeeze(0)
            
            # Compute anomaly score (MSE between predicted and current positions)
            current_pos = x[:, :2]
            mse = np.mean((predictions - current_pos) ** 2)
            
            return predictions, float(mse)
            
        except Exception as e:
            logger.warning(f"STGNN inference error: {e}")
            return None, 0.0


# ==========================================================
# VISUALIZATION ENGINE
# ==========================================================
class CrowdVisualizer:
    """Visualize detections, graph structure, and anomaly alerts."""
    
    def __init__(self):
        """Initialize visualizer with color scheme."""
        self.colors = {
            "normal": (0, 255, 0),      # Green
            "warning": (0, 255, 255),   # Yellow
            "critical": (0, 0, 255)     # Red
        }
        
        logger.info("[INIT] Visualization Engine")
    
    def draw(
        self,
        frame: np.ndarray,
        detections: List[Tuple[float, float]],
        graph: Optional[Dict],
        anomaly_score: float,
        frame_info: str = ""
    ) -> np.ndarray:
        """
        Create complete visualization.
        
        Args:
            frame: Input BGR frame
            detections: List of (x, y) pixel coordinates
            graph: Graph dictionary
            anomaly_score: Computed anomaly score
            frame_info: Additional frame information
        
        Returns:
            Annotated frame
        """
        vis = frame.copy()
        
        # Determine alert level
        if anomaly_score > ANOMALY_THRESHOLD_CRITICAL:
            alert = "critical"
        elif anomaly_score > ANOMALY_THRESHOLD_WARNING:
            alert = "warning"
        else:
            alert = "normal"
        
        # Draw graph edges (spatial connections)
        if DISPLAY_GRAPH_EDGES and graph is not None:
            self._draw_edges(vis, detections, graph["edge_index"])
        
        # Draw person detections
        self._draw_detections(vis, detections, alert)
        
        # Draw information overlay
        self._draw_overlay(vis, len(detections), anomaly_score, alert, frame_info)
        
        return vis
    
    def _draw_edges(
        self,
        frame: np.ndarray,
        detections: List[Tuple[float, float]],
        edge_index: np.ndarray
    ):
        """Draw graph edges."""
        for src, dst in edge_index.T:
            if src < len(detections) and dst < len(detections):
                pt1 = (int(detections[src][0]), int(detections[src][1]))
                pt2 = (int(detections[dst][0]), int(detections[dst][1]))
                cv2.line(frame, pt1, pt2, (80, 80, 80), 1)
    
    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Tuple[float, float]],
        alert: str
    ):
        """Draw person detections with color-coded alert level."""
        color = self.colors[alert]
        
        for x, y in detections:
            # Inner filled circle
            cv2.circle(frame, (int(x), int(y)), 6, color, -1)
            # Outer ring
            cv2.circle(frame, (int(x), int(y)), 12, color, 2)
    
    def _draw_overlay(
        self,
        frame: np.ndarray,
        num_people: int,
        anomaly_score: float,
        alert: str,
        frame_info: str
    ):
        """Draw information overlay."""
        h, w = frame.shape[:2]
        
        # Semi-transparent background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (380, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(frame, f"People: {num_people}", (20, 50),
                   font, 0.9, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Anomaly: {anomaly_score:.5f}", (20, 90),
                   font, 0.7, (255, 255, 255), 2)
        
        alert_color = self.colors[alert]
        cv2.putText(frame, f"Status: {alert.upper()}", (20, 130),
                   font, 0.8, alert_color, 2)
        
        # Frame info (if provided)
        if frame_info:
            cv2.putText(frame, frame_info, (w - 220, h - 20),
                       font, 0.5, (200, 200, 200), 1)
        
        # Instructions
        cv2.putText(frame, "Q: Quit | SPACE: Pause", (20, h - 20),
                   font, 0.5, (180, 180, 180), 1)


# ==========================================================
# MAIN PIPELINE EXECUTION
# ==========================================================
def main():
    """Main pipeline execution."""
    try:
        # Validate setup
        validate_setup()
        
        logger.info("[STEP 1/4] Initializing components...")
        
        # Initialize YOLO detector
        yolo = YOLODetector(
            model_path=YOLO_MODEL,
            conf_threshold=0.10,
            iou_threshold=0.45,
            enable_gpu=True  # Changed to True for GPU
        )
        
        # Initialize graph builder
        graph_builder = RealtimeGraphBuilder(radius=GRAPH_RADIUS)
        
        # Initialize STGNN inference
        stgnn = STGNNInference(STGNN_MODEL)
        
        # Initialize visualizer
        visualizer = CrowdVisualizer()
        
        logger.info("[OK] All components initialized\n")
        
        # Setup video source
        logger.info("[STEP 2/4] Setting up video source...")
        
        if isinstance(VIDEO_SOURCE, str) and os.path.isdir(VIDEO_SOURCE):
            # Dataset mode
            frame_files = sorted([f for f in os.listdir(VIDEO_SOURCE) if f.endswith('.jpg')])
            frame_paths = [os.path.join(VIDEO_SOURCE, f) for f in frame_files]
            total_frames = len(frame_paths)
            logger.info(f"[OK] Dataset mode: {total_frames} frames\n")
        else:
            # Video/webcam mode
            cap = cv2.VideoCapture(VIDEO_SOURCE)
            if not cap.isOpened():
                logger.error(f"Failed to open video source: {VIDEO_SOURCE}")
                sys.exit(1)
            frame_paths = None
            
            # Get total frames for video
            if isinstance(VIDEO_SOURCE, str):
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            else:
                total_frames = None
            
            logger.info(f"[OK] Video/webcam mode")
            if total_frames:
                logger.info(f"     Total frames: {total_frames}\n")
        
        # =====================================================
        # SETUP OUTPUT VIDEO WRITER
        # =====================================================
        video_writer = None
        if SAVE_OUTPUT_VIDEO:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            # Read first frame to get dimensions
            if frame_paths is not None:
                first_frame = cv2.imread(frame_paths[0])
            else:
                ret, first_frame = cap.read()
                if ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
            
            if first_frame is not None:
                frame_h, frame_w = first_frame.shape[:2]
                
                # Create output filename with timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_video_path = os.path.join(OUTPUT_DIR, f"pipeline_output_{timestamp}.mp4")
                
                # Initialize video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_video_path, fourcc, OUTPUT_FPS, (frame_w, frame_h))
                
                logger.info(f"[OK] Output video: {output_video_path}")
                logger.info(f"     Resolution: {frame_w}x{frame_h} @ {OUTPUT_FPS} FPS\n")
        
        # Process frames
        logger.info("[STEP 3/4] Processing pipeline...")
        logger.info("Controls: Q=Quit, SPACE=Pause\n")
        
        frame_idx = 0
        paused = False
        prev_time = time.time()
        
        while True:
            # Read frame
            if frame_paths is not None:
                # Dataset mode
                if frame_idx >= len(frame_paths):
                    logger.info("\nReached end of dataset")
                    break
                
                if not paused:
                    frame = cv2.imread(frame_paths[frame_idx])
                    if frame is None:
                        logger.warning(f"Failed to read frame {frame_idx}")
                        frame_idx += 1
                        continue
                    
                    frame_info = f"Frame: {frame_idx+1}/{total_frames}"
            else:
                # Video/webcam mode
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        logger.info("\nVideo ended")
                        break
                    
                    if total_frames:
                        frame_info = f"Frame: {frame_idx+1}/{total_frames}"
                    else:
                        frame_info = f"Frame: {frame_idx+1}"
            
            if not paused:
                frame_h, frame_w = frame.shape[:2]
                
                # Stage 1: YOLO Detection
                centers, detections = yolo.detect_persons_with_boxes(frame)
                
                # Stage 2: Graph Building
                graph = graph_builder.build_graph(centers, (frame_h, frame_w))
                
                # Stage 3: STGNN Inference
                predictions, anomaly_score = stgnn.predict(graph)
                
                # Stage 4: Visualization
                vis = visualizer.draw(frame, centers, graph, anomaly_score, frame_info)
                
                # Add FPS display
                if DISPLAY_FPS:
                    curr_time = time.time()
                    fps = 1.0 / (curr_time - prev_time + 1e-6)
                    prev_time = curr_time
                    cv2.putText(vis, f"FPS: {fps:.1f}", (frame_w - 150, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Save to output video
                if video_writer is not None:
                    video_writer.write(vis)
                
                # Display
                cv2.imshow("Crowd Anomaly Detection", vis)
                
                frame_idx += 1
                
                # Progress indicator
                if frame_idx % 50 == 0:
                    if total_frames:
                        progress = (frame_idx / total_frames) * 100
                        logger.info(f"Progress: {frame_idx}/{total_frames} ({progress:.1f}%)")
            
            # Handle keyboard input
            key = cv2.waitKey(FRAME_DELAY if not paused else 0) & 0xFF
            
            if key == ord('q'):
                logger.info("\nUser quit")
                break
            elif key == ord(' '):
                paused = not paused
                logger.info(f"{'[PAUSED]' if paused else '[RESUMED]'}")
        
        # Cleanup
        if frame_paths is None:
            cap.release()
        
        # Release video writer
        if video_writer is not None:
            video_writer.release()
            logger.info(f"\n[OK] Output video saved: {output_video_path}")
            
            # Show file size
            if os.path.exists(output_video_path):
                file_size = os.path.getsize(output_video_path) / (1024 * 1024)
                logger.info(f"     File size: {file_size:.2f} MB")
        
        cv2.destroyAllWindows()
        
        # Final statistics
        logger.info("\n[STEP 4/4] Pipeline completed")
        logger.info("=" * 70)
        
        stats = yolo.get_stats()
        logger.info(f"Total frames processed: {stats['frames_processed']}")
        logger.info(f"Average FPS: {stats['fps']:.2f}")
        logger.info(f"Average inference time: {stats['avg_inference_time_ms']:.2f} ms")
        
        if SAVE_OUTPUT_VIDEO and os.path.exists(output_video_path):
            logger.info(f"\n[OUTPUT] Video saved to:")
            logger.info(f"         {output_video_path}")
        
        logger.info("=" * 70)
        logger.info("[SUCCESS] Pipeline finished successfully!")
        logger.info("=" * 70)
        
    except KeyboardInterrupt:
        logger.info("\n\n[INTERRUPTED] Pipeline stopped by user")
        if video_writer is not None:
            video_writer.release()
            logger.info(f"[OK] Partial output saved: {output_video_path}")
        cv2.destroyAllWindows()
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"\n[ERROR] Pipeline failed: {str(e)}")
        if video_writer is not None:
            video_writer.release()
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()
        sys.exit(1)


if __name__ == "__main__":
    main()

