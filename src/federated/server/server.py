"""
Federated Server
----------------
Central server orchestrating federated learning rounds.

Responsibilities:
- Coordinate federated rounds
- Handle device registration
- Accept model updates
- Trigger aggregation (min K devices OR timeout)
- Update global model
- Distribute new weights (PyTorch + ONNX path)

This implementation uses LOCAL (in-process) transport only.
Network transport can be added later without changing this interface.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import torch
import yaml

from ..protocol.messages import (
    RegisterDevice,
    SubmitUpdate,
    Heartbeat,
    RegisterAck,
    AggregatedModel,
    UpdateAck,
    StateDict,
)
from .model_manager import ModelManager
from .aggregator import Aggregator, AggregationResult
from .device_registry import DeviceRegistry, DeviceInfo, DeviceStatus


logger = logging.getLogger(__name__)


class RoundStatus(Enum):
    """Status of a federated round."""
    WAITING = "WAITING"           # Waiting for updates
    AGGREGATING = "AGGREGATING"   # Aggregation in progress
    DISTRIBUTING = "DISTRIBUTING" # Distributing results
    COMPLETE = "COMPLETE"         # Round complete


@dataclass
class ServerConfig:
    """
    Configuration for federated server.
    
    Attributes:
        min_clients: Minimum clients before aggregation.
        round_timeout_sec: Timeout for a round before forcing aggregation.
        stale_device_timeout_sec: Timeout before marking device stale.
        model_class: PyTorch model class.
        model_kwargs: Arguments for model instantiation.
        model_init_path: Optional path to initial weights.
        onnx_export_dir: Directory for ONNX exports.
    """
    min_clients: int = 2
    round_timeout_sec: float = 300.0
    stale_device_timeout_sec: float = 120.0
    model_class: type = None  # Must be set
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    model_init_path: Optional[str] = None
    onnx_export_dir: str = "outputs/federated/models"
    
    @classmethod
    def from_yaml(cls, yaml_path: str, model_class: type) -> "ServerConfig":
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration.
            model_class: PyTorch model class (cannot be serialized in YAML).
        
        Returns:
            ServerConfig instance.
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        return cls(
            min_clients=data.get("min_clients", 2),
            round_timeout_sec=data.get("round_timeout_sec", 300.0),
            stale_device_timeout_sec=data.get("stale_device_timeout_sec", 120.0),
            model_class=model_class,
            model_kwargs=data.get("model_kwargs", {}),
            model_init_path=data.get("model_init_path"),
            onnx_export_dir=data.get("onnx_export_dir", "outputs/federated/models"),
        )


@dataclass
class RoundInfo:
    """Information about the current or last round."""
    round_id: int
    status: RoundStatus
    started_at: float
    completed_at: Optional[float] = None
    participating_devices: List[str] = field(default_factory=list)
    total_samples: int = 0
    result_version: int = 0


class FederatedServer:
    """
    Central federated learning server.
    
    Orchestrates federated rounds:
    1. Devices register
    2. Devices submit updates (state_dict + sample count)
    3. When min_clients submit OR timeout:
       - Aggregate via FedAvg
       - Update global model
       - Export ONNX
       - Distribute to devices
    
    Uses LOCAL (in-process) transport. Devices call methods directly.
    
    Usage:
        server = FederatedServer(config)
        
        # Device registration
        ack = server.register_device(RegisterDevice(...))
        
        # Device submits update
        ack = server.submit_update(SubmitUpdate(...))
        
        # Check for aggregated model
        model = server.get_aggregated_model(device_id)
    """
    
    def __init__(self, config: ServerConfig):
        """
        Initialize federated server.
        
        Args:
            config: Server configuration.
        """
        if config.model_class is None:
            raise ValueError("model_class must be specified in config")
        
        self._config = config
        
        # Initialize components
        self._model_manager = ModelManager(
            model_class=config.model_class,
            model_kwargs=config.model_kwargs,
            initial_weights_path=config.model_init_path,
            onnx_export_dir=config.onnx_export_dir,
        )
        
        self._aggregator = Aggregator(
            expected_param_keys=self._model_manager.param_keys,
        )
        
        self._registry = DeviceRegistry(
            stale_timeout_sec=config.stale_device_timeout_sec,
        )
        
        # Round state
        self._current_round_id = 0
        self._round_started_at: Optional[float] = None
        self._round_status = RoundStatus.WAITING
        self._last_aggregated_model: Optional[AggregatedModel] = None
        
        # Pending model distributions (device_id -> AggregatedModel)
        self._pending_distributions: Dict[str, AggregatedModel] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Callback for when aggregation completes
        # Contract: Invoked ONLY AFTER all of the following:
        #   1. Aggregation completes successfully
        #   2. Global model version increments
        #   3. ONNX export succeeds (or fails gracefully)
        #   4. Aggregated model is queued for distribution
        self._on_aggregation_complete: Optional[Callable[[AggregatedModel], None]] = None
        
        # Background timeout watcher thread
        # Automatically checks for round timeout without external polling
        self._timeout_watcher_stop = threading.Event()
        self._timeout_watcher_interval = min(1.0, config.round_timeout_sec / 5.0)
        self._timeout_watcher_thread: Optional[threading.Thread] = None
        
        # Start the timeout watcher
        self._start_timeout_watcher()
        
        logger.info(
            "FederatedServer initialized: min_clients=%d, timeout=%.0fs, watcher_interval=%.2fs",
            config.min_clients,
            config.round_timeout_sec,
            self._timeout_watcher_interval,
        )
    
    # ============================================================
    # Device Registration
    # ============================================================
    
    def register_device(self, msg: RegisterDevice) -> RegisterAck:
        """
        Handle device registration.
        
        Args:
            msg: Registration message from device.
        
        Returns:
            Registration acknowledgement.
        """
        with self._lock:
            try:
                device = self._registry.register(
                    device_id=msg.device_id,
                    device_type=msg.device_type,
                    model_version=msg.current_model_version,
                )
                
                # If device is behind, queue model distribution
                if msg.current_model_version < self._model_manager.version:
                    self._queue_distribution(msg.device_id)
                
                return RegisterAck(
                    device_id=msg.device_id,
                    success=True,
                    current_global_version=self._model_manager.version,
                )
                
            except Exception as exc:
                logger.error("Registration failed for %s: %s", msg.device_id, exc)
                return RegisterAck(
                    device_id=msg.device_id,
                    success=False,
                    error_message=str(exc),
                )
    
    # ============================================================
    # Update Submission
    # ============================================================
    
    def submit_update(self, msg: SubmitUpdate) -> UpdateAck:
        """
        Handle model update submission from a device.
        
        Args:
            msg: Update message containing state_dict.
        
        Returns:
            Update acknowledgement.
        """
        with self._lock:
            # Validate device is registered
            device = self._registry.get(msg.device_id)
            if device is None:
                return UpdateAck(
                    device_id=msg.device_id,
                    success=False,
                    error_message="Device not registered",
                )
            
            # Validate version compatibility
            current_version = self._model_manager.version
            if msg.base_version != current_version:
                logger.warning(
                    "Version mismatch from %s: base=%d, current=%d",
                    msg.device_id, msg.base_version, current_version,
                )
                # Still accept but log warning - device may be behind
            
            # Start round if first update
            if self._round_started_at is None:
                self._start_round()
            
            # Add to aggregator
            accepted = self._aggregator.add_update(
                device_id=msg.device_id,
                state_dict=msg.state_dict,
                num_samples=msg.num_samples,
                base_version=msg.base_version,
            )
            
            if not accepted:
                return UpdateAck(
                    device_id=msg.device_id,
                    success=False,
                    round_id=self._current_round_id,
                    error_message="Update rejected (invalid state_dict)",
                )
            
            # Record in registry
            self._registry.record_update_submission(
                device_id=msg.device_id,
                sample_count=msg.num_samples,
                model_version=msg.base_version,
            )
            
            logger.info(
                "Received update from %s: %d samples (round %d, %d/%d clients)",
                msg.device_id,
                msg.num_samples,
                self._current_round_id,
                self._aggregator.update_count,
                self._config.min_clients,
            )
            
            # Check if we should trigger aggregation
            self._check_aggregation_trigger()
            
            return UpdateAck(
                device_id=msg.device_id,
                success=True,
                round_id=self._current_round_id,
            )
    
    # ============================================================
    # Heartbeat
    # ============================================================
    
    def handle_heartbeat(self, msg: Heartbeat) -> bool:
        """
        Handle heartbeat from device.
        
        Args:
            msg: Heartbeat message.
        
        Returns:
            True if heartbeat was processed.
        """
        with self._lock:
            return self._registry.update_heartbeat(
                device_id=msg.device_id,
                model_version=msg.current_model_version,
                sample_count=msg.sample_count,
            )
    
    # ============================================================
    # Model Distribution
    # ============================================================
    
    def get_aggregated_model(self, device_id: str) -> Optional[AggregatedModel]:
        """
        Get pending aggregated model for a device.
        
        This is used by devices to poll for new models.
        
        Args:
            device_id: Device requesting the model.
        
        Returns:
            AggregatedModel if one is pending, None otherwise.
        """
        with self._lock:
            return self._pending_distributions.pop(device_id, None)
    
    def get_current_model(self) -> AggregatedModel:
        """
        Get the current global model.
        
        Useful for new devices that need the latest model.
        
        Returns:
            AggregatedModel with current weights.
        """
        with self._lock:
            return AggregatedModel(
                version=self._model_manager.version,
                state_dict=self._model_manager.get_state_dict(),
                onnx_path=self._model_manager.latest_onnx_path or "",
                participating_devices=0,
                total_samples=0,
            )
    
    # ============================================================
    # Round Management
    # ============================================================
    
    def _start_round(self) -> None:
        """Start a new federated round."""
        self._current_round_id += 1
        self._round_started_at = time.time()
        self._round_status = RoundStatus.WAITING
        self._aggregator.clear()
        
        logger.info("Started federated round %d", self._current_round_id)
    
    def _check_aggregation_trigger(self) -> None:
        """Check if aggregation should be triggered."""
        if self._round_status != RoundStatus.WAITING:
            return
        
        # Trigger if enough clients
        if self._aggregator.update_count >= self._config.min_clients:
            logger.info(
                "Triggering aggregation: %d clients (min=%d)",
                self._aggregator.update_count,
                self._config.min_clients,
            )
            self._perform_aggregation()
    
    def check_round_timeout(self) -> bool:
        """
        Check if round has timed out and trigger aggregation if so.
        
        This is called automatically by the internal timeout watcher thread.
        Can also be called manually for testing purposes.
        
        Returns:
            True if aggregation was triggered due to timeout.
        
        Note:
            This method is thread-safe and acquires the server lock.
        """
        with self._lock:
            if self._round_status != RoundStatus.WAITING:
                return False
            
            if self._round_started_at is None:
                return False
            
            elapsed = time.time() - self._round_started_at
            
            if elapsed >= self._config.round_timeout_sec:
                if self._aggregator.update_count > 0:
                    logger.info(
                        "Round timeout reached (%.0fs), triggering aggregation with %d clients",
                        elapsed,
                        self._aggregator.update_count,
                    )
                    self._perform_aggregation()
                    return True
                else:
                    # No updates, just reset
                    logger.info("Round timeout with no updates, resetting")
                    self._round_started_at = None
                    self._round_status = RoundStatus.WAITING
            
            return False
    
    def _start_timeout_watcher(self) -> None:
        """
        Start the background timeout watcher thread.
        
        The watcher periodically checks for round timeout and triggers
        aggregation automatically without requiring external polling.
        
        Internal API: Used by __init__ and tests.
        """
        if self._timeout_watcher_thread is not None:
            return  # Already running
        
        self._timeout_watcher_stop.clear()
        self._timeout_watcher_thread = threading.Thread(
            target=self._timeout_watcher_loop,
            name="FederatedServer-TimeoutWatcher",
            daemon=True,
        )
        self._timeout_watcher_thread.start()
        logger.debug("Timeout watcher started")
    
    def _stop_timeout_watcher(self) -> None:
        """
        Stop the background timeout watcher thread.
        
        Waits for the watcher thread to terminate gracefully.
        
        Internal API: Used by shutdown() and tests.
        """
        self._timeout_watcher_stop.set()
        
        if self._timeout_watcher_thread is not None:
            self._timeout_watcher_thread.join(timeout=2.0)
            self._timeout_watcher_thread = None
        
        logger.debug("Timeout watcher stopped")
    
    def _timeout_watcher_loop(self) -> None:
        """
        Background loop that periodically checks for round timeout.
        
        Runs until _timeout_watcher_stop is set.
        """
        while not self._timeout_watcher_stop.is_set():
            try:
                self.check_round_timeout()
            except Exception as exc:
                logger.error("Timeout watcher error: %s", exc)
            
            # Wait for interval or until stop is signaled
            self._timeout_watcher_stop.wait(timeout=self._timeout_watcher_interval)
    
    def _perform_aggregation(self) -> None:
        """Perform FedAvg aggregation."""
        self._round_status = RoundStatus.AGGREGATING
        
        result = self._aggregator.aggregate(
            expected_version=self._model_manager.version,
            min_clients=1,  # We already checked min_clients
        )
        
        if not result.success:
            logger.error("Aggregation failed: %s", result.error_message)
            self._round_status = RoundStatus.WAITING
            return
        
        # Update global model
        try:
            new_version = self._model_manager.update_weights(
                result.aggregated_state_dict,
            )
        except Exception as exc:
            logger.error("Failed to update global model: %s", exc)
            self._round_status = RoundStatus.WAITING
            return
        
        # Export ONNX
        try:
            onnx_path = self._model_manager.export_onnx()
        except Exception as exc:
            logger.error("ONNX export failed: %s", exc)
            onnx_path = ""
        
        # Create aggregated model message
        aggregated = AggregatedModel(
            version=new_version,
            state_dict=self._model_manager.get_state_dict(),
            onnx_path=onnx_path,
            participating_devices=result.num_clients,
            total_samples=result.total_samples,
        )
        
        self._last_aggregated_model = aggregated
        
        # Distribute to participating devices
        self._round_status = RoundStatus.DISTRIBUTING
        self._distribute_model(aggregated, result.participating_devices)
        
        # Update registry versions
        self._registry.update_model_version(
            result.participating_devices,
            new_version,
        )
        
        # Complete round
        self._round_status = RoundStatus.COMPLETE
        self._round_started_at = None
        
        logger.info(
            "Round %d complete: v%d, %d clients, %d samples",
            self._current_round_id,
            new_version,
            result.num_clients,
            result.total_samples,
        )
        
        # Call callback if set
        if self._on_aggregation_complete:
            try:
                self._on_aggregation_complete(aggregated)
            except Exception as exc:
                logger.error("Aggregation callback failed: %s", exc)
    
    def _distribute_model(
        self,
        model: AggregatedModel,
        device_ids: List[str],
    ) -> None:
        """Queue model distribution to devices."""
        for device_id in device_ids:
            self._pending_distributions[device_id] = model
    
    def _queue_distribution(self, device_id: str) -> None:
        """Queue current model for distribution to a device."""
        if self._last_aggregated_model is not None:
            self._pending_distributions[device_id] = self._last_aggregated_model
    
    # ============================================================
    # Status & Callbacks
    # ============================================================
    
    def set_aggregation_callback(
        self,
        callback: Callable[[AggregatedModel], None],
    ) -> None:
        """
        Set callback for when aggregation completes.
        
        The callback is invoked ONLY AFTER all of the following:
        1. FedAvg aggregation completes successfully
        2. Global model version is incremented
        3. ONNX export succeeds (or fails gracefully with empty path)
        4. Aggregated model is queued for distribution to all participants
        
        The callback receives the final AggregatedModel with:
        - version: New global model version
        - state_dict: Aggregated weights
        - onnx_path: Path to exported ONNX file (may be empty on export failure)
        - participating_devices: Number of devices in this round
        - total_samples: Total samples used in aggregation
        
        Args:
            callback: Function called with AggregatedModel.
        
        Note:
            The callback is invoked from within the server lock.
            It should be lightweight and not block.
        """
        self._on_aggregation_complete = callback
    
    def get_round_info(self) -> RoundInfo:
        """Get information about current/last round."""
        with self._lock:
            return RoundInfo(
                round_id=self._current_round_id,
                status=self._round_status,
                started_at=self._round_started_at or 0,
                participating_devices=self._aggregator.get_device_ids(),
                total_samples=self._aggregator.total_samples,
                result_version=self._model_manager.version,
            )
    
    def get_stats(self) -> dict:
        """Get server statistics."""
        with self._lock:
            return {
                "round_id": self._current_round_id,
                "round_status": self._round_status.value,
                "model_version": self._model_manager.version,
                "pending_updates": self._aggregator.update_count,
                "pending_distributions": len(self._pending_distributions),
                "registry": self._registry.get_stats(),
                "model_info": {
                    "param_count": self._model_manager.get_info().param_count,
                    "onnx_path": self._model_manager.latest_onnx_path,
                },
            }
    
    @property
    def model_version(self) -> int:
        """Current global model version."""
        return self._model_manager.version
    
    @property
    def registry(self) -> DeviceRegistry:
        """Get device registry."""
        return self._registry
    
    # ============================================================
    # Administrative
    # ============================================================
    
    def force_aggregation(self) -> bool:
        """
        Force aggregation with whatever updates are available.
        
        Returns:
            True if aggregation was performed.
        """
        with self._lock:
            if self._aggregator.update_count == 0:
                return False
            
            self._perform_aggregation()
            return True
    
    def mark_stale_devices(self) -> List[str]:
        """Mark stale devices in registry."""
        with self._lock:
            return self._registry.mark_stale_devices()
    
    def shutdown(self) -> None:
        """
        Shutdown the server cleanly.
        
        Stops the background timeout watcher thread.
        Should be called when the server is no longer needed.
        
        Note:
            After shutdown, the server should not be used.
            Create a new instance if you need a server again.
        """
        logger.info("Shutting down FederatedServer...")
        self._stop_timeout_watcher()
        logger.info("FederatedServer shutdown complete")
