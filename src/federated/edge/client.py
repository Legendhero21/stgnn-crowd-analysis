"""
Edge Client
-----------
Production-grade edge device abstraction for federated STGNN.

This module wraps the existing real-time pipeline WITHOUT modifying it.
All existing components (YOLO, GraphBuilder, TemporalBuffer, Metrics, Alerts)
are used as-is.

Key responsibilities:
- Orchestrate the existing pipeline components
- Use ONNX for inference (via ONNXHotSwapper)
- Collect training samples (via TrainingBuffer)
- Expose clean output interface for future dashboard
- Support dynamic model updates from federated server
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

# Add parent to path for imports from existing code
_src_dir = Path(__file__).parent.parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Import existing components (unchanged)
from alert_logic import StampedeAlert
from crowd_metrics import CrowdMetrics
from temporal_buffer import TemporalGraphBuffer
from yolo_detector import YOLODetector

# Import federated components
from .config import EdgeConfig
from .graph_builder import GraphBuilder
from .video_source import VideoSource, create_video_source, FrameData
from .onnx_swapper import ONNXHotSwapper
from .training_buffer import TrainingBuffer


logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """
    Result from processing a single frame.
    
    Contains all information needed for visualization and analytics.
    """
    frame_idx: int
    timestamp_ms: float
    
    # Detection results
    centers: List[Tuple[float, float]]  # Person centers in pixels
    num_persons: int
    
    # Graph data
    graph: Optional[Dict[str, np.ndarray]]  # {x, edge_index} or None
    
    # Inference results
    anomaly_score: float
    
    # Metrics from CrowdMetrics
    metrics: Dict[str, float]
    
    # Alert state
    alert_state: str  # "NORMAL", "UNSTABLE", "STAMPEDE"
    
    # Model info
    model_version: int
    
    # Processing time
    processing_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "frame_idx": self.frame_idx,
            "timestamp_ms": self.timestamp_ms,
            "num_persons": self.num_persons,
            "anomaly_score": self.anomaly_score,
            "metrics": self.metrics,
            "alert_state": self.alert_state,
            "model_version": self.model_version,
            "processing_time_ms": self.processing_time_ms,
        }



class EdgeClient:
    """
    Edge device client for federated STGNN crowd analysis.
    
    Wraps the existing pipeline with:
    - ONNX-based inference (hot-swappable)
    - Training sample collection
    - Clean output interface
    
    Usage:
        config = EdgeConfig(...)
        client = EdgeClient(config)
        client.start()
        
        # Processing runs in background
        # Results available via callback or get_latest_result()
        
        client.stop()
    """
    
    def __init__(
        self,
        config: EdgeConfig,
        result_callback: Optional[Callable[[FrameResult], None]] = None,
    ):
        """
        Initialize edge client.
        
        Args:
            config: Edge device configuration.
            result_callback: Optional callback for each frame result.
        """
        self._config = config
        self._result_callback = result_callback
        
        # Generate device ID if not provided
        self._device_id = config.device_id or self._generate_device_id()
        
        # Components
        self._video_source: Optional[VideoSource] = None
        self._yolo: Optional[YOLODetector] = None
        self._graph_builder: Optional[GraphBuilder] = None
        self._temporal_buffer: Optional[TemporalGraphBuffer] = None
        self._onnx_swapper: Optional[ONNXHotSwapper] = None
        self._alert_logic: Optional[StampedeAlert] = None
        self._training_buffer: Optional[TrainingBuffer] = None
        
        # Visualization (optional)
        self._video_writer: Optional[cv2.VideoWriter] = None
        
        # State
        self._is_running = False
        self._is_initialized = False
        self._run_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Results
        self._latest_result: Optional[FrameResult] = None
        self._latest_frame: Optional[np.ndarray] = None  # raw BGR for streaming
        self._frame_count = 0
        self._start_time: Optional[float] = None
        
        # Delayed graph for training sample creation
        # Tracks previous temporal sequence and node count for sample pairing
        self._previous_x_seq: Optional[np.ndarray] = None
        self._previous_edge_index: Optional[np.ndarray] = None
        self._previous_node_count: int = 0  # Guard against node count changes
        self._previous_frame_idx: int = 0
        
        logger.info("EdgeClient created: device_id=%s", self._device_id)
    
    def _generate_device_id(self) -> str:
        """Generate a unique device ID."""
        import uuid
        import platform
        
        # Combine hostname + random UUID for uniqueness
        hostname = platform.node()[:8] if platform.node() else "edge"
        short_uuid = str(uuid.uuid4())[:8]
        return f"{hostname}-{short_uuid}"
    
    def initialize(self) -> bool:
        """
        Initialize all components.
        
        Returns:
            True if initialization succeeded.
        """
        if self._is_initialized:
            logger.warning("EdgeClient already initialized")
            return True
        
        try:
            logger.info("Initializing EdgeClient...")
            
            # 1. Video source
            self._video_source = create_video_source(
                self._config.video_source,
                loop=True,  # Loop for continuous operation
            )
            
            if not self._video_source.open():
                raise RuntimeError(f"Failed to open video source: {self._config.video_source}")
            
            logger.info("Video source initialized: %s", self._video_source.source_id)
            
            # 2. YOLO detector
            self._yolo = YOLODetector(
                model_path=self._config.yolo_model_path,
                conf_threshold=self._config.yolo_conf_threshold,
                device=self._config.yolo_device,
            )
            
            logger.info("YOLO detector initialized")
            
            # 3. Graph builder
            self._graph_builder = GraphBuilder(
                radius=self._config.graph_radius,
                min_nodes=self._config.min_nodes,
            )
            
            # 4. Temporal buffer
            self._temporal_buffer = TemporalGraphBuffer(
                window_size=self._config.temporal_window,
            )
            
            # 5. ONNX inference (hot-swappable)
            self._onnx_swapper = ONNXHotSwapper(
                initial_model_path=self._config.stgnn_onnx_path,
                initial_version=0,
            )
            
            logger.info("ONNX inference initialized: %s", self._onnx_swapper.info)
            
            # 6. Alert logic
            self._alert_logic = StampedeAlert()
            
            # 7. Training buffer
            self._training_buffer = TrainingBuffer(
                max_samples=self._config.training_buffer_size,
            )
            
            # 8. Video writer (optional)
            if self._config.save_output_video:
                self._setup_video_writer()
            
            self._is_initialized = True
            logger.info("EdgeClient initialization complete")
            
            return True
            
        except Exception as exc:
            logger.error("EdgeClient initialization failed: %s", exc)
            self.cleanup()
            return False
    
    def _setup_video_writer(self) -> None:
        """Setup video writer for output."""
        if self._video_source is None:
            return
        
        os.makedirs(self._config.output_dir, exist_ok=True)
        
        w, h = self._video_source.frame_size
        if w <= 0 or h <= 0:
            logger.warning("Invalid frame size, skipping video writer")
            return
        
        output_path = os.path.join(
            self._config.output_dir,
            f"edge_{self._device_id}_output.mp4",
        )
        
        self._video_writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self._config.output_fps,
            (w, h),
        )
        
        logger.info("Video writer initialized: %s", output_path)
    
    def start(self, blocking: bool = False) -> None:
        """
        Start processing loop.
        
        Args:
            blocking: If True, run in current thread. Otherwise, start background thread.
        """
        if not self._is_initialized:
            if not self.initialize():
                raise RuntimeError("EdgeClient initialization failed")
        
        if self._is_running:
            logger.warning("EdgeClient already running")
            return
        
        self._is_running = True
        self._start_time = time.time()
        
        if blocking:
            self._run_loop()
        else:
            self._run_thread = threading.Thread(
                target=self._run_loop,
                name=f"EdgeClient-{self._device_id}",
                daemon=True,
            )
            self._run_thread.start()
            logger.info("EdgeClient started in background thread")
    
    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop processing loop.
        
        Args:
            timeout: Timeout for thread join.
        """
        logger.info("Stopping EdgeClient...")
        self._is_running = False
        
        if self._run_thread is not None and self._run_thread.is_alive():
            self._run_thread.join(timeout=timeout)
        
        self.cleanup()
        logger.info("EdgeClient stopped")
    
    def cleanup(self) -> None:
        """Release all resources."""
        if self._video_source is not None:
            self._video_source.close()
        
        if self._video_writer is not None:
            self._video_writer.release()
        
        if self._config.display_visualization:
            cv2.destroyAllWindows()
    
    def _run_loop(self) -> None:
        """Main processing loop."""
        logger.info("Processing loop started")
        
        try:
            while self._is_running and self._video_source is not None:
                frame_data = self._video_source.read()
                
                if frame_data is None:
                    if not self._video_source.is_open():
                        logger.info("Video source closed, stopping loop")
                        break
                    continue
                
                result = self._process_frame(frame_data)
                
                if result is not None:
                    with self._lock:
                        self._latest_result = result
                        self._frame_count += 1
                    
                    if self._result_callback is not None:
                        try:
                            self._result_callback(result)
                        except Exception as exc:
                            logger.error("Result callback failed: %s", exc)
                    
                    # Visualization
                    if self._config.display_visualization:
                        self._visualize(frame_data.frame, result)
                
                # Check for exit key
                if self._config.display_visualization:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("User requested exit")
                        break
                
        except Exception as exc:
            logger.error("Processing loop error: %s", exc)
        
        finally:
            self._is_running = False
            logger.info("Processing loop ended (frames: %d)", self._frame_count)
    
    def _process_frame(self, frame_data: FrameData) -> Optional[FrameResult]:
        """
        Process a single frame through the pipeline.
        
        Args:
            frame_data: Input frame data.
        
        Returns:
            FrameResult or None if processing failed.
        """
        start_time = time.time()
        
        frame = frame_data.frame
        frame_idx = frame_data.frame_idx
        
        # 1. YOLO detection
        try:
            centers, _ = self._yolo.detect_persons_with_boxes(frame)
            centers = centers or []
        except Exception as exc:
            logger.error("YOLO detection failed: %s", exc)
            centers = []
        
        # 2. Graph building
        graph = self._graph_builder.build_graph(centers, frame.shape[:2])
        
        # 3. Temporal buffering
        x_seq = self._temporal_buffer.push(graph)
        
        # 4. ONNX inference
        if x_seq is not None and graph is not None:
            anomaly_score = self._onnx_swapper.predict_from_sequence(
                x_seq, graph["edge_index"]
            )
        else:
            anomaly_score = 0.0
        
        # 5. Training sample collection (delayed by one frame)
        # Determine current node count for stability check
        current_node_count = graph["x"].shape[0] if graph is not None else 0
        
        self._collect_training_sample(graph, current_node_count, frame_idx)
        
        # Update delayed state for next frame ONLY if:
        # - We have valid data
        # - Node count is stable (matches previous or this is first valid frame)
        if x_seq is not None and graph is not None:
            if self._previous_node_count == 0 or current_node_count == self._previous_node_count:
                # Node count stable, safe to store
                self._previous_x_seq = x_seq
                self._previous_edge_index = graph["edge_index"]
                self._previous_frame_idx = frame_idx
                self._previous_node_count = current_node_count
            else:
                # Node count changed - discard previous state to avoid pairing mismatch
                logger.debug(
                    "Node count changed (%d -> %d), discarding training state",
                    self._previous_node_count,
                    current_node_count,
                )
                self._previous_x_seq = None
                self._previous_edge_index = None
                self._previous_node_count = 0
        
        # 6. Crowd metrics
        metrics = CrowdMetrics.compute(graph)
        
        # 7. Alert logic
        alert_state = self._alert_logic.update(anomaly_score, metrics)
        
        # 8. Draw visualization overlay (always, for streaming)
        vis_frame = self._draw_visualization(frame, centers, graph, anomaly_score, alert_state)
        self._latest_frame = vis_frame
        
        # Video writing (optional)
        if self._video_writer is not None:
            self._video_writer.write(vis_frame)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return FrameResult(
            frame_idx=frame_idx,
            timestamp_ms=frame_data.timestamp_ms,
            centers=centers,
            num_persons=len(centers),
            graph=graph,
            anomaly_score=anomaly_score,
            metrics=metrics,
            alert_state=alert_state,
            model_version=self._onnx_swapper.version,
            processing_time_ms=processing_time_ms,
        )
    
    def _collect_training_sample(
        self,
        current_graph: Optional[dict],
        current_node_count: int,
        frame_idx: int,
    ) -> None:
        """
        Collect training sample using previous x_seq and current positions.
        
        The target is the current frame's positions, used to train on
        predicting next-frame positions from the previous temporal sequence.
        
        Guards:
        - Previous x_seq and edge_index must exist
        - Current graph must be valid
        - Node count must match between previous and current
        """
        # Guard: no previous state
        if self._previous_x_seq is None or self._previous_edge_index is None:
            return
        
        # Guard: no current graph
        if current_graph is None or "x" not in current_graph:
            logger.debug("No current graph, skipping training sample")
            return
        
        # Guard: node count mismatch (TemporalBuffer may have reset)
        prev_n_nodes = self._previous_x_seq.shape[2]
        if prev_n_nodes != current_node_count:
            logger.debug(
                "Node count mismatch (prev=%d, curr=%d), skipping training sample",
                prev_n_nodes,
                current_node_count,
            )
            return
        
        current_features = current_graph["x"]
        current_positions = current_features[:, :2]  # [N, 2]
        
        # Add sample
        self._training_buffer.add(
            x_seq=self._previous_x_seq,
            edge_index=self._previous_edge_index,
            next_frame_positions=current_positions,
            frame_idx=self._previous_frame_idx,
        )
    
    def _draw_visualization(
        self,
        frame: np.ndarray,
        centers: List[Tuple[float, float]],
        graph: Optional[dict],
        anomaly: float,
        state: str,
    ) -> np.ndarray:
        """Draw visualization overlay on frame."""
        vis = frame.copy()
        
        # Determine color based on alert state
        colors = {
            "NORMAL": (0, 255, 0),
            "UNSTABLE": (0, 255, 255),
            "STAMPEDE": (0, 0, 255),
        }
        color = colors.get(state, (255, 255, 255))
        
        # Draw edges
        if graph is not None and len(centers) > 1:
            edge_index = graph.get("edge_index")
            if edge_index is not None and edge_index.size > 0:
                for s, d in edge_index.T:
                    if 0 <= s < len(centers) and 0 <= d < len(centers):
                        p1 = tuple(map(int, centers[s]))
                        p2 = tuple(map(int, centers[d]))
                        cv2.line(vis, p1, p2, (80, 80, 80), 1)
        
        # Draw centers
        for x, y in centers:
            cv2.circle(vis, (int(x), int(y)), 5, color, -1)
        
        # Draw text overlay
        cv2.putText(
            vis,
            f"People: {len(centers)} | Anomaly: {anomaly:.5f} | {state}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )
        
        cv2.putText(
            vis,
            f"Device: {self._device_id} | Model v{self._onnx_swapper.version}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        
        # FPS
        if self._start_time is not None and self._frame_count > 0:
            elapsed = time.time() - self._start_time
            fps = self._frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(
                vis,
                f"FPS: {fps:.1f}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
        
        return vis
    
    def _visualize(self, frame: np.ndarray, result: FrameResult) -> None:
        """Display visualization in window."""
        vis = self._draw_visualization(
            frame,
            result.centers,
            result.graph,
            result.anomaly_score,
            result.alert_state,
        )
        
        window_name = f"EdgeClient: {self._device_id}"
        cv2.imshow(window_name, vis)
    
    # ====================
    # Public API
    # ====================
    
    @property
    def device_id(self) -> str:
        """Get device ID."""
        return self._device_id
    
    @property
    def is_running(self) -> bool:
        """Check if client is running."""
        return self._is_running
    
    @property
    def frame_count(self) -> int:
        """Get total processed frames."""
        with self._lock:
            return self._frame_count
    
    @property
    def training_buffer(self) -> Optional[TrainingBuffer]:
        """Get training buffer for local training."""
        return self._training_buffer
    
    @property
    def model_version(self) -> int:
        """Get current STGNN model version."""
        return self._onnx_swapper.version if self._onnx_swapper else 0
    
    @property
    def is_initialized(self) -> bool:
        """Check if client has been initialized."""
        return self._is_initialized
    
    def replace_training_buffer(self, new_buffer) -> None:
        """
        Replace the training buffer.
        
        Used for testing/simulation to inject synthetic data.
        
        Args:
            new_buffer: New TrainingBuffer-like object.
        """
        self._training_buffer = new_buffer
    
    def get_latest_result(self) -> Optional[FrameResult]:
        """Get the most recent frame result."""
        with self._lock:
            return self._latest_result
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the most recent raw video frame (BGR) for streaming."""
        with self._lock:
            return self._latest_frame
    
    def get_latest_result(self) -> Optional[FrameResult]:
        """Get the most recent processing result for dashboard metrics."""
        with self._lock:
            return self._latest_result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        with self._lock:
            elapsed = time.time() - self._start_time if self._start_time else 0
            fps = self._frame_count / elapsed if elapsed > 0 else 0
            
            return {
                "device_id": self._device_id,
                "is_running": self._is_running,
                "frame_count": self._frame_count,
                "elapsed_sec": elapsed,
                "fps": fps,
                "model_version": self._onnx_swapper.version if self._onnx_swapper else 0,
                "training_buffer": self._training_buffer.get_stats() if self._training_buffer else None,
                "yolo_stats": self._yolo.get_stats() if self._yolo else None,
            }
    
    def update_model(self, new_onnx_path: str, new_version: int) -> bool:
        """
        Hot-swap the STGNN model.
        
        Args:
            new_onnx_path: Path to new ONNX model.
            new_version: New version number.
        
        Returns:
            True if swap succeeded.
        """
        if self._onnx_swapper is None:
            logger.error("Cannot update model: not initialized")
            return False
        
        return self._onnx_swapper.hot_swap(new_onnx_path, new_version)
    
    def update_onnx_model(self, new_model_path: str, new_version: int) -> bool:
        """
        Alias for update_model (used by FederatedClient._apply_aggregated_model).
        
        Args:
            new_model_path: Path to new ONNX model.
            new_version: New version number.
        
        Returns:
            True if swap succeeded.
        """
        return self.update_model(new_model_path, new_version)
