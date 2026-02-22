"""
Dashboard Adapter
-----------------
READ-ONLY observability adapter for the federated learning system.

This adapter mirrors state from the LOCKED core system (Phase 1-4).
It does NOT:
- Generate analytics
- Perform inference
- Run training
- Modify system state
- Access private attributes

It ONLY reads from public APIs:
- FederatedServer.get_stats()
- FederatedClient.get_stats()
- EdgeClient.get_latest_result()

Usage:
    adapter = DashboardAdapter(server, clients)
    metrics = adapter.get_all_edge_metrics()  # Returns List[EdgeMetrics]
    status = adapter.get_server_status()      # Returns ServerStatus
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .schemas import (
    AlertState,
    ClientState,
    DeviceInfo,
    DeviceStatus,
    EdgeMetrics,
    ModelInfo,
    RegistryStats,
    ServerStatus,
    TrainingStatus,
)

if TYPE_CHECKING:
    # Avoid circular imports - these are from the core system
    # The adapter receives instances at runtime
    pass


logger = logging.getLogger(__name__)


class DashboardAdapter:
    """
    Read-only adapter that mirrors state from the federated system.
    
    INVARIANTS:
    - Never modifies any core system state
    - Never accesses private attributes
    - Never raises exceptions that crash the backend
    - Never blocks the core system
    
    Thread Safety:
    - All core system methods already acquire their own locks
    - Adapter does not hold additional locks
    - Reads are atomic snapshots
    
    Usage:
        # At startup
        server = FederatedServer(config)
        clients = {"edge_01": FederatedClient(...), ...}
        adapter = DashboardAdapter(server, clients)
        
        # During operation
        metrics = adapter.get_all_edge_metrics()
    """
    
    def __init__(
        self,
        server: Any,  # FederatedServer
        clients: Optional[Dict[str, Any]] = None,  # Dict[str, FederatedClient]
    ):
        """
        Initialize dashboard adapter.
        
        Args:
            server: FederatedServer instance (READ-ONLY access).
            clients: Dict mapping device_id to FederatedClient (READ-ONLY access).
        """
        self._server = server
        self._clients: Dict[str, Any] = clients or {}
        
        logger.info(
            "DashboardAdapter initialized: server=%s, clients=%d",
            type(server).__name__ if server else None,
            len(self._clients),
        )
    
    # ================================================================
    # CLIENT MANAGEMENT (for dynamic registration)
    # ================================================================
    
    def register_client(self, device_id: str, client: Any) -> None:
        """
        Register a FederatedClient for observation.
        
        Called when a new client joins the system.
        
        Args:
            device_id: Unique device identifier.
            client: FederatedClient instance.
        """
        self._clients[device_id] = client
        logger.debug("Registered client for observation: %s", device_id)
    
    def unregister_client(self, device_id: str) -> None:
        """
        Unregister a FederatedClient.
        
        Called when a client leaves the system.
        
        Args:
            device_id: Device to unregister.
        """
        self._clients.pop(device_id, None)
        logger.debug("Unregistered client: %s", device_id)
    
    @property
    def device_ids(self) -> List[str]:
        """List of observed device IDs."""
        return list(self._clients.keys())
    
    def get_edge_client(self, device_id: str) -> Optional[Any]:
        """
        Get the EdgeClient instance for a given device ID.
        
        Used by the video streaming endpoint to access raw frames.
        
        Args:
            device_id: Device identifier.
            
        Returns:
            EdgeClient instance or None if not found.
        """
        client = self._clients.get(device_id)
        if client is None:
            return None
        # FederatedClient wraps EdgeClient as _edge_client
        return getattr(client, '_edge_client', None)
    
    # ================================================================
    # EDGE METRICS (from EdgeClient)
    # ================================================================
    
    def get_edge_metrics(self, device_id: str) -> Optional[EdgeMetrics]:
        """
        Get metrics for a specific edge device.
        
        Source: EdgeClient.get_latest_result() via FederatedClient
        
        Args:
            device_id: Device to query.
            
        Returns:
            EdgeMetrics or None if unavailable.
        """
        client = self._clients.get(device_id)
        if client is None:
            return None
        
        try:
            # Access edge_client via FederatedClient's internal attribute
            edge_client = getattr(client, '_edge_client', None)
            
            if edge_client is None:
                # Fallback: client doesn't expose edge_client
                # Use client stats which include some edge info
                return self._edge_metrics_from_client_stats(device_id, client)
            
            result = edge_client.get_latest_result()
            if result is None:
                return None
            
            # Extract crowd metrics from the result
            metrics = result.metrics if hasattr(result, 'metrics') else {}
            
            return EdgeMetrics(
                device_id=device_id,
                timestamp_ms=result.timestamp_ms,
                frame_idx=result.frame_idx,
                num_persons=result.num_persons,
                anomaly_score=result.anomaly_score,
                alert_state=AlertState(result.alert_state),
                model_version=result.model_version,
                crowd_density=metrics.get("mean_density", 0.0),
                avg_velocity=metrics.get("mean_speed", 0.0),
                flow_magnitude=metrics.get("motion_entropy", 0.0),
                processing_time_ms=result.processing_time_ms,
            )
            
        except Exception as exc:
            logger.warning(
                "Failed to get edge metrics for %s: %s",
                device_id, exc,
            )
            return None
    
    def _edge_metrics_from_client_stats(
        self,
        device_id: str,
        client: Any,
    ) -> Optional[EdgeMetrics]:
        """
        Fallback: construct minimal EdgeMetrics from client stats.
        
        Used when edge_client is not publicly accessible.
        """
        try:
            stats = client.get_stats()
            return EdgeMetrics(
                device_id=device_id,
                timestamp_ms=time.time() * 1000,
                frame_idx=0,
                num_persons=0,
                anomaly_score=0.0,
                alert_state=AlertState.NORMAL,
                model_version=stats.get("model_version", 0),
                crowd_density=0.0,
                avg_velocity=0.0,
                flow_magnitude=0.0,
                processing_time_ms=0.0,
            )
        except Exception:
            return None
    
    def get_all_edge_metrics(self) -> List[EdgeMetrics]:
        """
        Get metrics for all observed edge devices.
        
        Returns:
            List of EdgeMetrics for devices with available data.
        """
        results = []
        for device_id in self._clients:
            metrics = self.get_edge_metrics(device_id)
            if metrics is not None:
                results.append(metrics)
        return results
    
    # ================================================================
    # TRAINING STATUS (from FederatedClient)
    # ================================================================
    
    def get_training_status(self, device_id: str) -> Optional[TrainingStatus]:
        """
        Get training status for a specific device.
        
        Source: FederatedClient.get_stats()
        
        Args:
            device_id: Device to query.
            
        Returns:
            TrainingStatus or None if unavailable.
        """
        client = self._clients.get(device_id)
        if client is None:
            return None
        
        try:
            stats = client.get_stats()
            
            return TrainingStatus(
                device_id=stats.get("device_id", device_id),
                state=ClientState(stats.get("state", "IDLE")),
                model_version=stats.get("model_version", 0),
                is_registered=stats.get("is_registered", False),
                training_rounds=stats.get("training_rounds", 0),
                samples_trained=stats.get("samples_trained", 0),
                samples_buffered=stats.get("samples_buffered", 0),
                last_training_time=stats.get("last_training_time"),
            )
            
        except Exception as exc:
            logger.warning(
                "Failed to get training status for %s: %s",
                device_id, exc,
            )
            return None
    
    def get_all_training_status(self) -> List[TrainingStatus]:
        """
        Get training status for all devices.
        
        Returns:
            List of TrainingStatus for all observed clients.
        """
        results = []
        for device_id in self._clients:
            status = self.get_training_status(device_id)
            if status is not None:
                results.append(status)
        return results
    
    # ================================================================
    # SERVER STATUS (from FederatedServer)
    # ================================================================
    
    def get_server_status(self) -> Optional[ServerStatus]:
        """
        Get federated server status.
        
        Source: FederatedServer.get_stats()
        
        Returns:
            ServerStatus or None if server unavailable.
        """
        if self._server is None:
            return None
        
        try:
            stats = self._server.get_stats()
            
            # Parse nested registry stats
            registry_data = stats.get("registry", {})
            registry = RegistryStats(
                total_devices=registry_data.get("total_devices", 0),
                active_devices=registry_data.get("active_devices", 0),
                stale_devices=registry_data.get("stale_devices", 0),
                stale_timeout_sec=registry_data.get("stale_timeout_sec", 30.0),
            )
            
            # Parse nested model info
            model_data = stats.get("model_info", {})
            model_info = ModelInfo(
                param_count=model_data.get("param_count", 0),
                onnx_path=model_data.get("onnx_path"),
            )
            
            return ServerStatus(
                round_id=stats.get("round_id", 0),
                round_status=stats.get("round_status", "WAITING"),
                model_version=stats.get("model_version", 0),
                pending_updates=stats.get("pending_updates", 0),
                pending_distributions=stats.get("pending_distributions", 0),
                registry=registry,
                model_info=model_info,
            )
            
        except Exception as exc:
            logger.warning("Failed to get server status: %s", exc)
            return None
    
    # ================================================================
    # DEVICE LIST (from DeviceRegistry via Server)
    # ================================================================
    
    def get_device_list(self) -> List[DeviceInfo]:
        """
        Get list of all registered devices from server.
        
        Source: FederatedServer internal registry
        
        Note: This requires server to expose registry access.
        If not available, returns empty list.
        """
        if self._server is None:
            return []
        
        try:
            # Check if server exposes registry
            get_devices = getattr(self._server, 'get_registered_devices', None)
            if get_devices is not None:
                devices_data = get_devices()
            else:
                # Fallback: use stats registry info
                stats = self._server.get_stats()
                # Can't get individual devices, return empty
                return []
            
            results = []
            for device in devices_data:
                results.append(DeviceInfo(
                    device_id=device.device_id,
                    status=DeviceStatus(device.status.value),
                    model_version=device.model_version,
                    last_seen=device.last_seen,
                    registered_at=device.registered_at,
                ))
            return results
            
        except Exception as exc:
            logger.warning("Failed to get device list: %s", exc)
            return []
    
    # ================================================================
    # AGGREGATED SNAPSHOT
    # ================================================================
    
    def get_full_snapshot(self) -> Dict[str, Any]:
        """
        Get complete system snapshot.
        
        Returns dict with:
        - server: ServerStatus
        - devices: List[DeviceInfo]
        - edge_metrics: List[EdgeMetrics]
        - training_status: List[TrainingStatus]
        
        All data is READ-ONLY and safe for serialization.
        """
        return {
            "server": self.get_server_status(),
            "devices": self.get_device_list(),
            "edge_metrics": self.get_all_edge_metrics(),
            "training_status": self.get_all_training_status(),
            "timestamp": time.time(),
        }
