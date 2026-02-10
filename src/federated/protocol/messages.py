"""
Federated Protocol Messages
---------------------------
Message types for federated learning communication.

All messages are dataclasses that are pickle-safe for serialization.
Designed for local (in-process) transport, extensible to network transport.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# Type alias for PyTorch state_dict
StateDict = Dict[str, Any]


@dataclass
class Message:
    """
    Base class for all federated messages.
    
    Attributes:
        message_id: Unique identifier (auto-generated).
        timestamp: Unix timestamp when message was created.
    """
    message_id: str = field(default_factory=lambda: f"msg-{time.time_ns()}")
    timestamp: float = field(default_factory=time.time)
    
    def serialize(self) -> bytes:
        """Serialize message to bytes using pickle."""
        return pickle.dumps(self)
    
    @classmethod
    def deserialize(cls, data: bytes) -> "Message":
        """Deserialize message from bytes."""
        obj = pickle.loads(data)
        if not isinstance(obj, Message):
            raise ValueError(f"Expected Message, got {type(obj)}")
        return obj


# ============================================================
# Edge → Server Messages
# ============================================================

@dataclass
class RegisterDevice(Message):
    """
    Edge device registration request.
    
    Sent when an edge device joins the federated system.
    
    Attributes:
        device_id: Unique identifier for the edge device.
        device_type: Category (drone, raspi, laptop, mobile, server).
        current_model_version: Model version the device currently has.
    """
    device_id: str = ""
    device_type: str = "laptop"
    current_model_version: int = 0
    
    def __post_init__(self):
        if not self.device_id:
            raise ValueError("device_id is required")


@dataclass
class SubmitUpdate(Message):
    """
    Model update submission from edge device.
    
    Contains the locally trained model weights to be aggregated.
    This is the ONLY federated artifact - no video, graphs, or features.
    
    Attributes:
        device_id: Identifier of submitting device.
        base_version: Model version the update is based on.
        state_dict: PyTorch model state_dict (weights only).
        num_samples: Number of training samples used for this update.
    """
    device_id: str = ""
    base_version: int = 0
    state_dict: Optional[StateDict] = None
    num_samples: int = 0
    
    def __post_init__(self):
        if not self.device_id:
            raise ValueError("device_id is required")
        if self.state_dict is None:
            raise ValueError("state_dict is required")
        if self.num_samples < 0:
            raise ValueError("num_samples must be >= 0")
    
    def validate_state_dict(self, expected_keys: set) -> bool:
        """
        Validate state_dict has expected keys.
        
        Args:
            expected_keys: Set of expected parameter names.
        
        Returns:
            True if keys match.
        """
        if self.state_dict is None:
            return False
        return set(self.state_dict.keys()) == expected_keys


@dataclass
class Heartbeat(Message):
    """
    Heartbeat from edge device to server.
    
    Indicates device is still active and reports current state.
    
    Attributes:
        device_id: Identifier of the device.
        current_model_version: Model version in use.
        sample_count: Total samples in training buffer.
        is_training: Whether device is currently training.
    """
    device_id: str = ""
    current_model_version: int = 0
    sample_count: int = 0
    is_training: bool = False
    
    def __post_init__(self):
        if not self.device_id:
            raise ValueError("device_id is required")


# ============================================================
# Server → Edge Messages
# ============================================================

@dataclass
class RegisterAck(Message):
    """
    Acknowledgement of device registration.
    
    Attributes:
        device_id: Registered device ID.
        success: Whether registration succeeded.
        current_global_version: Current global model version.
        error_message: Error message if registration failed.
    """
    device_id: str = ""
    success: bool = True
    current_global_version: int = 0
    error_message: str = ""


@dataclass
class AggregatedModel(Message):
    """
    Distributed aggregated model from server to edges.
    
    Sent after FedAvg aggregation completes.
    
    Attributes:
        version: New global model version (monotonically increasing).
        state_dict: Aggregated PyTorch state_dict.
        onnx_path: Path to exported ONNX model for inference.
        participating_devices: Number of devices in this round.
        total_samples: Total samples used in aggregation.
    """
    version: int = 0
    state_dict: Optional[StateDict] = None
    onnx_path: str = ""
    participating_devices: int = 0
    total_samples: int = 0
    
    def __post_init__(self):
        if self.version < 0:
            raise ValueError("version must be >= 0")


@dataclass
class UpdateAck(Message):
    """
    Acknowledgement of update submission.
    
    Attributes:
        device_id: Device that submitted the update.
        success: Whether submission was accepted.
        round_id: Current round identifier.
        error_message: Error message if submission failed.
    """
    device_id: str = ""
    success: bool = True
    round_id: int = 0
    error_message: str = ""


# ============================================================
# Utility Functions
# ============================================================

def create_register_device(
    device_id: str,
    device_type: str = "laptop",
    current_version: int = 0,
) -> RegisterDevice:
    """Factory function for RegisterDevice message."""
    return RegisterDevice(
        device_id=device_id,
        device_type=device_type,
        current_model_version=current_version,
    )


def create_submit_update(
    device_id: str,
    state_dict: StateDict,
    num_samples: int,
    base_version: int,
) -> SubmitUpdate:
    """Factory function for SubmitUpdate message."""
    return SubmitUpdate(
        device_id=device_id,
        base_version=base_version,
        state_dict=state_dict,
        num_samples=num_samples,
    )


def create_heartbeat(
    device_id: str,
    model_version: int,
    sample_count: int = 0,
    is_training: bool = False,
) -> Heartbeat:
    """Factory function for Heartbeat message."""
    return Heartbeat(
        device_id=device_id,
        current_model_version=model_version,
        sample_count=sample_count,
        is_training=is_training,
    )
