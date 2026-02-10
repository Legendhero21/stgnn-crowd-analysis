"""
Dashboard Backend Package
-------------------------
READ-ONLY observability backend for federated learning.

This package provides:
- DashboardAdapter: Read-only mirror of system state
- Pydantic schemas: Type-safe API responses
- FastAPI app: REST + WebSocket endpoints

Usage:
    from dashboard_backend.adapter import DashboardAdapter
    from dashboard_backend.main import app, set_adapter
"""

from .adapter import DashboardAdapter
from .schemas import (
    AlertState,
    ClientState,
    DeviceInfo,
    DeviceStatus,
    EdgeMetrics,
    RoundStatus,
    ServerStatus,
    TrainingStatus,
)

__all__ = [
    "DashboardAdapter",
    "AlertState",
    "ClientState", 
    "DeviceInfo",
    "DeviceStatus",
    "EdgeMetrics",
    "RoundStatus",
    "ServerStatus",
    "TrainingStatus",
]
