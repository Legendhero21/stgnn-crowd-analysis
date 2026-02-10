"""
Dashboard Backend
-----------------
FastAPI backend for real-time federated learning observability.

This is a READ-ONLY backend that mirrors state from the core system.
It does NOT:
- Generate analytics
- Perform inference
- Run training
- Modify system state

Data Flow:
    FederatedServer → DashboardAdapter → FastAPI → WebSocket → Frontend
                ↑
    FederatedClient(s)

Usage:
    # Option 1: Standalone (requires adapter injection)
    uvicorn main:app --reload
    
    # Option 2: With integration script (recommended)
    python run_with_dashboard.py
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .adapter import DashboardAdapter
from .schemas import (
    BatchUpdate,
    DashboardUpdate,
    EdgeMetrics,
    ServerStatus,
    TrainingStatus,
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# GLOBAL ADAPTER (injected at startup)
# ============================================================

# The adapter is injected by the integration script
# or set manually before starting the server
_adapter: Optional[DashboardAdapter] = None


def set_adapter(adapter: DashboardAdapter) -> None:
    """
    Inject the dashboard adapter.
    
    Must be called BEFORE starting the FastAPI server.
    
    Args:
        adapter: Configured DashboardAdapter instance.
    """
    global _adapter
    _adapter = adapter
    logger.info("Dashboard adapter injected")


def get_adapter() -> DashboardAdapter:
    """
    Get the injected adapter.
    
    Raises:
        RuntimeError: If adapter not injected.
    """
    if _adapter is None:
        raise RuntimeError(
            "Dashboard adapter not injected. "
            "Call set_adapter() before starting server."
        )
    return _adapter


# ============================================================
# WEBSOCKET CONNECTION MANAGER
# ============================================================

class ConnectionManager:
    """
    Manages WebSocket connections for real-time updates.
    
    Thread-safe for asyncio context.
    """
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new connection."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(
            "WebSocket connected. Total: %d",
            len(self.active_connections),
        )
    
    async def disconnect(self, websocket: WebSocket) -> None:
        """Unregister a connection."""
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(
            "WebSocket disconnected. Total: %d",
            len(self.active_connections),
        )
    
    async def broadcast_json(self, data: Dict[str, Any]) -> None:
        """
        Broadcast JSON data to all connected clients.
        
        Handles disconnected clients gracefully.
        """
        disconnected = []
        
        async with self._lock:
            connections = list(self.active_connections)
        
        for connection in connections:
            try:
                await connection.send_json(data)
            except Exception:
                disconnected.append(connection)
        
        # Clean up disconnected
        if disconnected:
            async with self._lock:
                for conn in disconnected:
                    if conn in self.active_connections:
                        self.active_connections.remove(conn)
    
    @property
    def connection_count(self) -> int:
        """Number of active connections."""
        return len(self.active_connections)


manager = ConnectionManager()


# ============================================================
# BACKGROUND BROADCASTER
# ============================================================

_broadcast_task: Optional[asyncio.Task] = None
_broadcast_stop = asyncio.Event()


async def broadcast_loop(interval_sec: float = 1.0) -> None:
    """
    Background task that broadcasts system state.
    
    Runs continuously, sending snapshots to all WebSocket clients.
    
    Args:
        interval_sec: Broadcast interval in seconds.
    """
    logger.info("Starting broadcast loop (interval=%.1fs)", interval_sec)
    
    while not _broadcast_stop.is_set():
        try:
            if manager.connection_count > 0:
                adapter = get_adapter()
                
                # Get current state
                server_status = adapter.get_server_status()
                edge_metrics = adapter.get_all_edge_metrics()
                training_status = adapter.get_all_training_status()
                
                # Build batch update
                updates = []
                
                if server_status:
                    updates.append(DashboardUpdate(
                        type="server_status",
                        data=server_status.dict(),
                    ))
                
                for metrics in edge_metrics:
                    updates.append(DashboardUpdate(
                        type="edge_metrics",
                        data=metrics.dict(),
                    ))
                
                for status in training_status:
                    updates.append(DashboardUpdate(
                        type="training_status",
                        data=status.dict(),
                    ))
                
                # Broadcast
                batch = BatchUpdate(
                    updates=updates,
                    timestamp=time.time(),
                )
                await manager.broadcast_json(batch.dict())
                
        except RuntimeError:
            # Adapter not available yet
            pass
        except Exception as exc:
            logger.warning("Broadcast error: %s", exc)
        
        # Wait for next interval or stop signal
        try:
            await asyncio.wait_for(
                _broadcast_stop.wait(),
                timeout=interval_sec,
            )
            break  # Stop signal received
        except asyncio.TimeoutError:
            pass  # Continue broadcasting


# ============================================================
# FASTAPI APP LIFECYCLE
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.
    
    Starts/stops background broadcast task.
    """
    global _broadcast_task
    
    # Startup
    _broadcast_stop.clear()
    _broadcast_task = asyncio.create_task(broadcast_loop())
    logger.info("Dashboard backend started")
    
    yield
    
    # Shutdown
    _broadcast_stop.set()
    if _broadcast_task:
        _broadcast_task.cancel()
        try:
            await _broadcast_task
        except asyncio.CancelledError:
            pass
    logger.info("Dashboard backend stopped")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Federated STGNN Dashboard",
    description="READ-ONLY observability backend for federated learning system",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REST ENDPOINTS
# ============================================================

@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "adapter_ready": _adapter is not None,
        "websocket_connections": manager.connection_count,
    }


@app.get("/api/server", response_model=Optional[ServerStatus])
def get_server_status():
    """
    Get federated server status.
    
    Returns:
        ServerStatus with round info, model version, registry stats.
    """
    try:
        adapter = get_adapter()
        status = adapter.get_server_status()
        if status is None:
            return JSONResponse(
                status_code=503,
                content={"error": "Server status unavailable"},
            )
        return status
    except RuntimeError as exc:
        return JSONResponse(
            status_code=503,
            content={"error": str(exc)},
        )


@app.get("/api/devices", response_model=List[EdgeMetrics])
def get_devices():
    """
    Get metrics for all edge devices.
    
    Returns:
        List of EdgeMetrics from all observed devices.
    """
    try:
        adapter = get_adapter()
        return adapter.get_all_edge_metrics()
    except RuntimeError as exc:
        return JSONResponse(
            status_code=503,
            content={"error": str(exc)},
        )


@app.get("/api/devices/{device_id}", response_model=Optional[EdgeMetrics])
def get_device_metrics(device_id: str):
    """
    Get metrics for a specific device.
    
    Args:
        device_id: Device identifier.
        
    Returns:
        EdgeMetrics for the device.
    """
    try:
        adapter = get_adapter()
        metrics = adapter.get_edge_metrics(device_id)
        if metrics is None:
            return JSONResponse(
                status_code=404,
                content={"error": f"Device not found: {device_id}"},
            )
        return metrics
    except RuntimeError as exc:
        return JSONResponse(
            status_code=503,
            content={"error": str(exc)},
        )


@app.get("/api/training", response_model=List[TrainingStatus])
def get_training_status():
    """
    Get training status for all devices.
    
    Returns:
        List of TrainingStatus for all observed clients.
    """
    try:
        adapter = get_adapter()
        return adapter.get_all_training_status()
    except RuntimeError as exc:
        return JSONResponse(
            status_code=503,
            content={"error": str(exc)},
        )


@app.get("/api/training/{device_id}", response_model=Optional[TrainingStatus])
def get_device_training(device_id: str):
    """
    Get training status for a specific device.
    
    Args:
        device_id: Device identifier.
        
    Returns:
        TrainingStatus for the device.
    """
    try:
        adapter = get_adapter()
        status = adapter.get_training_status(device_id)
        if status is None:
            return JSONResponse(
                status_code=404,
                content={"error": f"Device not found: {device_id}"},
            )
        return status
    except RuntimeError as exc:
        return JSONResponse(
            status_code=503,
            content={"error": str(exc)},
        )


@app.get("/api/snapshot")
def get_snapshot():
    """
    Get complete system snapshot.
    
    Returns:
        Full snapshot including server, devices, and training status.
    """
    try:
        adapter = get_adapter()
        return adapter.get_full_snapshot()
    except RuntimeError as exc:
        return JSONResponse(
            status_code=503,
            content={"error": str(exc)},
        )


# ============================================================
# WEBSOCKET ENDPOINT
# ============================================================

@app.websocket("/ws/analytics")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates.
    
    Clients connect and receive BatchUpdate messages automatically.
    No client-to-server messages are expected.
    """
    await manager.connect(websocket)
    try:
        # Send initial snapshot
        try:
            adapter = get_adapter()
            snapshot = adapter.get_full_snapshot()

            # Convert snapshot to pure JSON
            if isinstance(snapshot, dict):
                safe_snapshot = {}
                for key, value in snapshot.items():
                    if isinstance(value, list):
                        safe_snapshot[key] = [
                            v.dict() if hasattr(v, "dict") else v
                            for v in value
                        ]
                    elif hasattr(value, "dict"):
                        safe_snapshot[key] = value.dict()
                    else:
                        safe_snapshot[key] = value
            else:
                safe_snapshot = snapshot

            await websocket.send_json({
                "type": "initial_snapshot",
                "data": safe_snapshot,
            })
        except RuntimeError:
            await websocket.send_json({
                "type": "error",
                "data": {"message": "Adapter not ready"},
            })
        
        # Keep connection alive, broadcast loop handles updates
        while True:
            try:
                # We don't expect client messages, but need to keep alive
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
                
    finally:
        await manager.disconnect(websocket)


# ============================================================
# STANDALONE ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.warning(
        "Running standalone without adapter. "
        "Use integration script for real data."
    )
    
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
