"""
Edge Device Configuration
-------------------------
Production-grade configuration for federated edge devices.

All configuration is explicit and documented. No hidden defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal


@dataclass(frozen=True)
class EdgeConfig:
    """
    Immutable configuration for an edge device in the federated system.
    
    All paths are validated at construction time.
    All thresholds have explicit, documented values.
    
    Attributes:
        device_id: Unique identifier for this edge device. If None, auto-generated.
        device_type: Category of device (for logging/analytics).
        
        video_source: Path to video file OR stream URL (rtsp://, http://).
        yolo_model_path: Path to YOLOv11n .pt model file.
        stgnn_onnx_path: Path to STGNN ONNX model for inference.
        stgnn_pytorch_path: Path to STGNN PyTorch weights for training.
        
        output_dir: Directory for outputs (videos, logs, temp ONNX files).
        
        graph_radius: Radius for spatial graph edges (normalized coords).
        min_nodes: Minimum nodes required to build a graph.
        temporal_window: Number of frames in temporal buffer.
        
        yolo_conf_threshold: YOLO detection confidence threshold.
        yolo_device: Device for YOLO inference ("cuda" or "cpu").
        
        anomaly_threshold_warning: Anomaly score for WARNING state.
        anomaly_threshold_critical: Anomaly score for CRITICAL state.
        
        training_buffer_size: Maximum samples in training buffer.
        
        display_visualization: Whether to show CV2 window.
        save_output_video: Whether to save annotated video.
        output_fps: FPS for output video.
    """
    
    # === Device Identity ===
    device_id: Optional[str] = None
    device_type: Literal["drone", "raspi", "laptop", "mobile", "server"] = "laptop"
    
    # === Paths ===
    video_source: str = ""
    yolo_model_path: str = ""
    stgnn_onnx_path: str = ""
    stgnn_pytorch_path: str = ""
    output_dir: str = ""
    
    # === Graph Parameters ===
    graph_radius: float = 0.05
    min_nodes: int = 2
    temporal_window: int = 5
    
    # === YOLO Parameters ===
    yolo_conf_threshold: float = 0.4
    yolo_device: Literal["cuda", "cpu"] = "cuda"
    
    # === Anomaly Thresholds ===
    anomaly_threshold_warning: float = 0.05
    anomaly_threshold_critical: float = 0.15
    
    # === Training Buffer ===
    training_buffer_size: int = 1000
    
    # === Visualization ===
    display_visualization: bool = True
    save_output_video: bool = True
    output_fps: int = 10
    
    def __post_init__(self) -> None:
        """Validate configuration at construction time."""
        errors = []
        
        # Validate paths exist
        if self.video_source and not self._is_stream_url(self.video_source):
            if not os.path.isfile(self.video_source):
                errors.append(f"video_source not found: {self.video_source}")
        
        if self.yolo_model_path and not os.path.isfile(self.yolo_model_path):
            errors.append(f"yolo_model_path not found: {self.yolo_model_path}")
        
        if self.stgnn_onnx_path and not os.path.isfile(self.stgnn_onnx_path):
            errors.append(f"stgnn_onnx_path not found: {self.stgnn_onnx_path}")
        
        # PyTorch path is optional (may be created from ONNX or server)
        if self.stgnn_pytorch_path and not os.path.isfile(self.stgnn_pytorch_path):
            # Just warn, don't fail - it may be created later
            pass
        
        # Validate numeric ranges
        if not (0.0 < self.graph_radius <= 1.0):
            errors.append(f"graph_radius must be in (0, 1], got {self.graph_radius}")
        
        if self.min_nodes < 1:
            errors.append(f"min_nodes must be >= 1, got {self.min_nodes}")
        
        if self.temporal_window < 2:
            errors.append(f"temporal_window must be >= 2, got {self.temporal_window}")
        
        if not (0.0 < self.yolo_conf_threshold < 1.0):
            errors.append(f"yolo_conf_threshold must be in (0, 1), got {self.yolo_conf_threshold}")
        
        if self.anomaly_threshold_warning <= 0:
            errors.append(f"anomaly_threshold_warning must be > 0, got {self.anomaly_threshold_warning}")
        
        if self.anomaly_threshold_critical <= self.anomaly_threshold_warning:
            errors.append(
                f"anomaly_threshold_critical ({self.anomaly_threshold_critical}) "
                f"must be > anomaly_threshold_warning ({self.anomaly_threshold_warning})"
            )
        
        if self.training_buffer_size < 10:
            errors.append(f"training_buffer_size must be >= 10, got {self.training_buffer_size}")
        
        if self.output_fps < 1:
            errors.append(f"output_fps must be >= 1, got {self.output_fps}")
        
        if errors:
            raise ValueError(f"EdgeConfig validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    @staticmethod
    def _is_stream_url(source: str) -> bool:
        """Check if source is a stream URL rather than file path."""
        return source.startswith(("rtsp://", "http://", "https://"))
    
    @classmethod
    def from_env(cls, prefix: str = "EDGE_") -> "EdgeConfig":
        """
        Create configuration from environment variables.
        
        Environment variables are prefixed with `prefix` (default: EDGE_).
        Example: EDGE_VIDEO_SOURCE, EDGE_YOLO_MODEL_PATH, etc.
        
        Args:
            prefix: Prefix for environment variable names.
        
        Returns:
            EdgeConfig instance populated from environment.
        """
        def get_env(name: str, default: str = "") -> str:
            return os.getenv(f"{prefix}{name}", default)
        
        def get_env_float(name: str, default: float) -> float:
            val = os.getenv(f"{prefix}{name}")
            return float(val) if val else default
        
        def get_env_int(name: str, default: int) -> int:
            val = os.getenv(f"{prefix}{name}")
            return int(val) if val else default
        
        def get_env_bool(name: str, default: bool) -> bool:
            val = os.getenv(f"{prefix}{name}")
            if val is None:
                return default
            return val.lower() in ("true", "1", "yes")
        
        return cls(
            device_id=get_env("DEVICE_ID") or None,
            device_type=get_env("DEVICE_TYPE", "laptop"),  # type: ignore
            video_source=get_env("VIDEO_SOURCE"),
            yolo_model_path=get_env("YOLO_MODEL_PATH"),
            stgnn_onnx_path=get_env("STGNN_ONNX_PATH"),
            stgnn_pytorch_path=get_env("STGNN_PYTORCH_PATH"),
            output_dir=get_env("OUTPUT_DIR"),
            graph_radius=get_env_float("GRAPH_RADIUS", 0.05),
            min_nodes=get_env_int("MIN_NODES", 2),
            temporal_window=get_env_int("TEMPORAL_WINDOW", 5),
            yolo_conf_threshold=get_env_float("YOLO_CONF_THRESHOLD", 0.4),
            yolo_device=get_env("YOLO_DEVICE", "cuda"),  # type: ignore
            anomaly_threshold_warning=get_env_float("ANOMALY_THRESHOLD_WARNING", 0.05),
            anomaly_threshold_critical=get_env_float("ANOMALY_THRESHOLD_CRITICAL", 0.15),
            training_buffer_size=get_env_int("TRAINING_BUFFER_SIZE", 1000),
            display_visualization=get_env_bool("DISPLAY_VISUALIZATION", True),
            save_output_video=get_env_bool("SAVE_OUTPUT_VIDEO", True),
            output_fps=get_env_int("OUTPUT_FPS", 10),
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "EdgeConfig":
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file.
        
        Returns:
            EdgeConfig instance.
        
        Raises:
            FileNotFoundError: If YAML file doesn't exist.
            ValueError: If YAML parsing fails.
        """
        import yaml
        
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        if not isinstance(data, dict):
            raise ValueError(f"YAML root must be a mapping, got {type(data)}")
        
        return cls(**data)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        from dataclasses import asdict
        return asdict(self)
    
    def with_overrides(self, **kwargs) -> "EdgeConfig":
        """
        Create a new config with specified fields overridden.
        
        Args:
            **kwargs: Fields to override.
        
        Returns:
            New EdgeConfig with overrides applied.
        """
        current = self.to_dict()
        current.update(kwargs)
        return EdgeConfig(**current)


# === Factory functions for common configurations ===

def create_default_config(
    video_source: str,
    yolo_model: str,
    stgnn_onnx: str,
    output_dir: str,
    device_id: Optional[str] = None,
) -> EdgeConfig:
    """
    Create a configuration with sensible defaults.
    
    Args:
        video_source: Path to video file or stream URL.
        yolo_model: Path to YOLO .pt model.
        stgnn_onnx: Path to STGNN ONNX model.
        output_dir: Output directory.
        device_id: Optional device ID (auto-generated if None).
    
    Returns:
        EdgeConfig with defaults applied.
    """
    return EdgeConfig(
        device_id=device_id,
        video_source=video_source,
        yolo_model_path=yolo_model,
        stgnn_onnx_path=stgnn_onnx,
        output_dir=output_dir,
    )


def create_simulation_config(
    video_source: str,
    base_dir: str = "D:/stgnn_project",
    device_id: Optional[str] = None,
) -> EdgeConfig:
    """
    Create configuration for simulation/testing.
    
    Uses default model paths from standard project structure.
    
    Args:
        video_source: Path to video file.
        base_dir: Project base directory.
        device_id: Optional device ID.
    
    Returns:
        EdgeConfig for simulation.
    """
    return EdgeConfig(
        device_id=device_id,
        device_type="laptop",
        video_source=video_source,
        yolo_model_path=os.path.join(base_dir, "models", "yolo11n_person_best.pt"),
        stgnn_onnx_path=os.path.join(base_dir, "outputs", "evaluation", "stgnn_final.onnx"),
        stgnn_pytorch_path=os.path.join(base_dir, "outputs", "checkpoints", "stgnn_latest.pt"),
        output_dir=os.path.join(base_dir, "outputs", "pipeline_results"),
        display_visualization=False,  # Headless for simulation
        save_output_video=False,
    )
