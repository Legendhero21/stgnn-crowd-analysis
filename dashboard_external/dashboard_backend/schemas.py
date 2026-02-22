"""
Dashboard Schemas
-----------------
Pydantic models for dashboard API responses.

These schemas mirror data from the LOCKED core system.
They are READ-ONLY representations - no analytics computation.

Alignment:
- EdgeMetrics ← EdgeClient.get_latest_result() (FrameResult)
- TrainingStatus ← FederatedClient.get_stats()
- ServerStatus ← FederatedServer.get_stats()
- DeviceInfo ← DeviceRegistry.get_all_devices()
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


# ============================================================
# ENUMS (aligned with core system)
# ============================================================

class AlertState(str, Enum):
    """
    Alert state from EdgeClient inference.
    
    Matches: src/federated/edge/client.py FrameResult.alert_state
    """
    NORMAL = "NORMAL"
    UNSTABLE = "UNSTABLE"
    STAMPEDE = "STAMPEDE"


class ClientState(str, Enum):
    """
    FederatedClient state.
    
    Matches: src/federated/client/federated_client.py ClientState
    """
    IDLE = "IDLE"
    REGISTERING = "REGISTERING"
    COLLECTING = "COLLECTING"
    TRAINING = "TRAINING"
    SUBMITTING = "SUBMITTING"
    WAITING_MODEL = "WAITING_MODEL"
    STOPPED = "STOPPED"


class RoundStatus(str, Enum):
    """
    Federated round status.
    
    Matches: src/federated/server/server.py RoundStatus
    """
    WAITING = "WAITING"
    AGGREGATING = "AGGREGATING"
    DISTRIBUTING = "DISTRIBUTING"
    COMPLETE = "COMPLETE"


class DeviceStatus(str, Enum):
    """
    Device status in registry.
    
    Matches: src/federated/server/device_registry.py DeviceStatus
    """
    ACTIVE = "ACTIVE"
    STALE = "STALE"
    OFFLINE = "OFFLINE"


# ============================================================
# EDGE METRICS (from EdgeClient.get_latest_result)
# ============================================================

class EdgeMetrics(BaseModel):
    """
    Real-time metrics from an EdgeClient.
    
    Source: EdgeClient.get_latest_result() → FrameResult
    
    This is a READ-ONLY snapshot of the edge device's latest frame.
    The dashboard backend does NOT compute these values.
    """
    device_id: str = Field(..., description="Unique device identifier")
    timestamp_ms: float = Field(..., description="Frame timestamp in milliseconds")
    frame_idx: int = Field(..., description="Frame sequence number")
    
    # Detection results
    num_persons: int = Field(..., description="Number of persons detected by YOLO")
    
    # Inference results (from ONNX STGNN)
    anomaly_score: float = Field(..., ge=0.0, le=1.0, description="STGNN anomaly score")
    alert_state: AlertState = Field(..., description="Alert classification")
    
    # Model info
    model_version: int = Field(..., ge=0, description="ONNX model version")
    
    # Crowd metrics (from CrowdMetrics)
    crowd_density: float = Field(0.0, description="Crowd density estimate")
    avg_velocity: float = Field(0.0, description="Average crowd velocity")
    flow_magnitude: float = Field(0.0, description="Optical flow magnitude")
    
    # Processing info
    processing_time_ms: float = Field(0.0, description="Frame processing time")
    
    class Config:
        use_enum_values = True


# ============================================================
# TRAINING STATUS (from FederatedClient.get_stats)
# ============================================================

class TrainingStatus(BaseModel):
    """
    Training status for a FederatedClient.
    
    Source: FederatedClient.get_stats()
    
    READ-ONLY snapshot of client's training state.
    """
    device_id: str = Field(..., description="Unique device identifier")
    state: ClientState = Field(..., description="Current client state")
    model_version: int = Field(..., ge=0, description="Model version in use")
    is_registered: bool = Field(..., description="Whether registered with server")
    
    # Training progress
    training_rounds: int = Field(0, ge=0, description="Completed training rounds")
    samples_trained: int = Field(0, ge=0, description="Total samples trained")
    samples_buffered: int = Field(0, ge=0, description="Samples in buffer")
    
    # Timing
    last_training_time: Optional[float] = Field(None, description="Last training timestamp")
    
    class Config:
        use_enum_values = True


# ============================================================
# SERVER STATUS (from FederatedServer.get_stats)
# ============================================================

class RegistryStats(BaseModel):
    """Nested registry statistics."""
    total_devices: int = Field(..., ge=0)
    active_devices: int = Field(..., ge=0)
    stale_devices: int = Field(..., ge=0)
    stale_timeout_sec: float = Field(..., gt=0)


class ModelInfo(BaseModel):
    """Nested model information."""
    param_count: int = Field(..., ge=0)
    onnx_path: Optional[str] = None


class ServerStatus(BaseModel):
    """
    Federated server status.
    
    Source: FederatedServer.get_stats()
    
    READ-ONLY snapshot of server's aggregation state.
    """
    round_id: int = Field(..., ge=0, description="Current round ID")
    round_status: RoundStatus = Field(..., description="Round status")
    model_version: int = Field(..., ge=0, description="Global model version")
    pending_updates: int = Field(..., ge=0, description="Pending client updates")
    pending_distributions: int = Field(..., ge=0, description="Pending model distributions")
    
    # Nested
    registry: RegistryStats = Field(..., description="Device registry stats")
    model_info: ModelInfo = Field(..., description="Model information")
    
    class Config:
        use_enum_values = True


# ============================================================
# DEVICE INFO (from DeviceRegistry.get_all_devices)
# ============================================================

class DeviceInfo(BaseModel):
    """
    Device information from registry.
    
    Source: DeviceRegistry.get_all_devices() → List[DeviceInfo dataclass]
    """
    device_id: str
    status: DeviceStatus
    model_version: int = Field(..., ge=0)
    last_seen: float
    registered_at: float
    
    class Config:
        use_enum_values = True


# ============================================================
# WEBSOCKET MESSAGE TYPES
# ============================================================

class DashboardUpdate(BaseModel):
    """
    WebSocket message wrapper.
    
    Sent by dashboard backend to all connected frontends.
    Type field determines how frontend should handle data.
    """
    type: Literal["edge_metrics", "training_status", "server_status", "device_list"]
    data: Union[EdgeMetrics, TrainingStatus, ServerStatus, List[DeviceInfo], Dict[str, Any]]
    
    class Config:
        use_enum_values = True


class BatchUpdate(BaseModel):
    """
    Batch of updates for efficiency.
    
    Sent when multiple updates should be delivered together.
    """
    updates: List[DashboardUpdate]
    timestamp: float = Field(..., description="Server-side timestamp")
