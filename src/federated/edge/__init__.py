# Federated STGNN - Edge Device Components
"""
This package provides edge device abstractions for the federated
STGNN crowd analysis system.
"""

from .config import EdgeConfig
from .graph_builder import GraphBuilder
from .video_source import VideoSource, VideoFileSource, VideoStreamSource
from .onnx_swapper import ONNXHotSwapper
from .training_buffer import TrainingBuffer, TrainingSample
from .client import EdgeClient

__all__ = [
    "EdgeConfig",
    "GraphBuilder",
    "VideoSource",
    "VideoFileSource",
    "VideoStreamSource",
    "ONNXHotSwapper",
    "TrainingBuffer",
    "TrainingSample",
    "EdgeClient",
]
