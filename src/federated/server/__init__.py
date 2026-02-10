# Federated Server
"""
Central server components for federated STGNN learning.

Provides:
- ModelManager: Global model storage and versioning
- Aggregator: FedAvg implementation
- DeviceRegistry: Edge device tracking
- FederatedServer: Round orchestration
"""

from .model_manager import ModelManager
from .aggregator import Aggregator, AggregationResult
from .device_registry import DeviceRegistry, DeviceInfo, DeviceStatus
from .server import FederatedServer, ServerConfig

__all__ = [
    "ModelManager",
    "Aggregator",
    "AggregationResult",
    "DeviceRegistry",
    "DeviceInfo",
    "DeviceStatus",
    "FederatedServer",
    "ServerConfig",
]
