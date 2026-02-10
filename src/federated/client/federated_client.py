"""
Federated Client
----------------
Client-side orchestrator for federated learning.

Responsibilities:
- Wrap an existing EdgeClient (composition, not inheritance)
- Handle device registration with FederatedServer
- Send periodic heartbeats
- Trigger local training using TrainingBuffer
- Submit updates (state_dict, num_samples, base_version)
- Receive aggregated models
- Trigger ONNX hot-swap on EdgeClient

The FederatedClient does NOT:
- Reimplement EdgeClient logic
- Perform inference (that's EdgeClient's job)
- Communicate directly with server (uses transport adapter)

Usage:
    edge_client = EdgeClient(edge_config)
    
    fed_client = FederatedClient(
        edge_client=edge_client,
        transport=LocalTransport(server),
        trainer=LocalTrainer(model_class=STGNN),
    )
    
    fed_client.start()  # Starts edge processing + federated loop
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

import torch.nn as nn
import yaml

from ..edge.client import EdgeClient
from ..transport.local_transport import TransportProtocol
from .local_trainer import LocalTrainer, TrainingResult


logger = logging.getLogger(__name__)


class FederatedClientState(Enum):
    """State of the federated client."""
    IDLE = "IDLE"
    REGISTERING = "REGISTERING"
    COLLECTING = "COLLECTING"
    TRAINING = "TRAINING"
    SUBMITTING = "SUBMITTING"
    STOPPED = "STOPPED"


@dataclass
class FederatedClientConfig:
    """
    Configuration for federated client.
    
    Attributes:
        training_interval_sec: Seconds between training attempts.
        heartbeat_interval_sec: Seconds between heartbeats.
        max_local_epochs: Maximum epochs per training round.
        min_samples_for_training: Minimum samples before training starts.
        learning_rate: Learning rate for local training.
        batch_size: Batch size for local training.
        model_class: PyTorch model class for training.
        model_kwargs: Arguments for model instantiation.
        device: PyTorch device (cpu or cuda).
    """
    training_interval_sec: float = 60.0
    heartbeat_interval_sec: float = 30.0
    max_local_epochs: int = 5
    min_samples_for_training: int = 32
    learning_rate: float = 0.001
    batch_size: int = 16
    model_class: Optional[Type[nn.Module]] = None
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    device: str = "cpu"
    
    @classmethod
    def from_yaml(cls, yaml_path: str, model_class: Type[nn.Module]) -> "FederatedClientConfig":
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration.
            model_class: PyTorch model class (cannot be serialized in YAML).
        
        Returns:
            FederatedClientConfig instance.
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        return cls(
            training_interval_sec=data.get("training_interval_sec", 60.0),
            heartbeat_interval_sec=data.get("heartbeat_interval_sec", 30.0),
            max_local_epochs=data.get("max_local_epochs", 5),
            min_samples_for_training=data.get("min_samples_for_training", 32),
            learning_rate=data.get("learning_rate", 0.001),
            batch_size=data.get("batch_size", 16),
            model_class=model_class,
            model_kwargs=data.get("model_kwargs", {}),
            device=data.get("device", "cpu"),
        )


class FederatedClient:
    """
    Client-side orchestrator for federated learning.
    
    Wraps an EdgeClient and adds federated learning capabilities:
    - Registration with server
    - Periodic heartbeats
    - Local training on collected samples
    - Model update submission
    - ONNX hot-swap when receiving new models
    
    Thread Model:
    - EdgeClient runs in its own thread (managed by EdgeClient)
    - FederatedClient runs a separate federated loop thread
    - Heartbeat runs in a separate timer thread
    
    Thread Safety:
    - Internal state is protected by locks
    - Transport is assumed to be thread-safe
    - EdgeClient operations are thread-safe
    
    Usage:
        fed_client = FederatedClient(edge_client, transport, trainer)
        fed_client.start()
        # ... processing runs ...
        fed_client.stop()
    """
    
    def __init__(
        self,
        edge_client: EdgeClient,
        transport: TransportProtocol,
        trainer: LocalTrainer,
        config: Optional[FederatedClientConfig] = None,
    ):
        """
        Initialize federated client.
        
        Args:
            edge_client: EdgeClient instance to wrap.
            transport: Transport adapter for server communication.
            trainer: LocalTrainer for edge-side training.
            config: Federated client configuration.
        """
        self._edge_client = edge_client
        self._transport = transport
        self._trainer = trainer
        self._config = config or FederatedClientConfig()
        
        # State
        self._state = FederatedClientState.IDLE
        self._is_running = False
        self._is_registered = False
        self._current_model_version = 0
        
        # Threads
        self._fed_loop_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self._training_rounds = 0
        self._samples_trained = 0
        self._last_training_time: Optional[float] = None
        
        logger.info(
            "FederatedClient created: device_id=%s, training_interval=%.1fs",
            self.device_id,
            self._config.training_interval_sec,
        )
    
    @property
    def device_id(self) -> str:
        """Get device ID from wrapped EdgeClient."""
        return self._edge_client.device_id
    
    @property
    def state(self) -> FederatedClientState:
        """Get current client state."""
        with self._lock:
            return self._state
    
    @property
    def model_version(self) -> int:
        """Get current model version."""
        with self._lock:
            return self._current_model_version
    
    @property
    def is_running(self) -> bool:
        """Check if client is running."""
        return self._is_running
    
    # ============================================================
    # Lifecycle
    # ============================================================
    
    def start(self, blocking: bool = False) -> bool:
        """
        Start the federated client.
        
        This will:
        1. Initialize and start the EdgeClient
        2. Register with the server
        3. Start the federated loop thread
        4. Start the heartbeat thread
        
        Args:
            blocking: If True, run federated loop in current thread.
        
        Returns:
            True if started successfully.
        """
        if self._is_running:
            logger.warning("FederatedClient already running")
            return True
        
        logger.info("Starting FederatedClient...")
        
        # Initialize edge client if needed
        if not self._edge_client.is_initialized:
            if not self._edge_client.initialize():
                logger.error("Failed to initialize EdgeClient")
                return False
        
        # Register with server
        if not self._register():
            logger.error("Failed to register with server")
            return False
        
        # Sync to latest model
        self._sync_model()
        
        # Start edge client processing
        self._edge_client.start(blocking=False)
        
        # Start federated loop
        self._is_running = True
        self._stop_event.clear()
        
        if blocking:
            self._run_federated_loop()
        else:
            self._fed_loop_thread = threading.Thread(
                target=self._run_federated_loop,
                name=f"FederatedClient-{self.device_id}",
                daemon=True,
            )
            self._fed_loop_thread.start()
            
            # Start heartbeat thread
            self._heartbeat_thread = threading.Thread(
                target=self._run_heartbeat_loop,
                name=f"Heartbeat-{self.device_id}",
                daemon=True,
            )
            self._heartbeat_thread.start()
        
        logger.info("FederatedClient started")
        return True
    
    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop the federated client.
        
        Args:
            timeout: Timeout for thread joins.
        """
        logger.info("Stopping FederatedClient...")
        
        self._is_running = False
        self._stop_event.set()
        
        # Stop heartbeat thread
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=timeout)
        
        # Stop federated loop thread
        if self._fed_loop_thread is not None and self._fed_loop_thread.is_alive():
            self._fed_loop_thread.join(timeout=timeout)
        
        # Stop edge client
        self._edge_client.stop(timeout=timeout)
        
        with self._lock:
            self._state = FederatedClientState.STOPPED
        
        logger.info("FederatedClient stopped")
    
    # ============================================================
    # Registration
    # ============================================================
    
    def _register(self) -> bool:
        """Register with the federated server."""
        with self._lock:
            self._state = FederatedClientState.REGISTERING
        
        ack = self._transport.register_device(
            device_id=self.device_id,
            device_type="laptop",
            current_model_version=self._current_model_version,
        )
        
        if ack.success:
            with self._lock:
                self._is_registered = True
                self._current_model_version = ack.current_global_version
            
            logger.info(
                "Registered with server: global_version=%d",
                ack.current_global_version,
            )
            return True
        else:
            logger.error("Registration failed: %s", ack.error_message)
            return False
    
    def _sync_model(self) -> None:
        """Sync to the latest global model."""
        current_model = self._transport.get_current_model()
        
        if current_model.version > self._current_model_version:
            self._apply_aggregated_model(current_model)
    
    # ============================================================
    # Federated Loop
    # ============================================================
    
    def _run_federated_loop(self) -> None:
        """
        Main federated learning loop.
        
        Periodically:
        1. Check for new aggregated models
        2. Train locally if enough samples
        3. Submit updates to server
        """
        logger.info("Federated loop started")
        
        last_check_time = time.time()
        
        while self._is_running and not self._stop_event.is_set():
            try:
                # Check for new aggregated model
                self._poll_for_model()
                
                # Check if it's time to train
                elapsed = time.time() - last_check_time
                
                if elapsed >= self._config.training_interval_sec:
                    self._training_cycle()
                    last_check_time = time.time()
                
                # Sleep before next iteration
                self._stop_event.wait(timeout=1.0)
                
            except Exception as exc:
                logger.error("Federated loop error: %s", exc)
                time.sleep(1.0)
        
        logger.info("Federated loop ended")
    
    def _run_heartbeat_loop(self) -> None:
        """Send periodic heartbeats to server."""
        logger.debug("Heartbeat loop started")
        
        while self._is_running and not self._stop_event.is_set():
            try:
                self._send_heartbeat()
            except Exception as exc:
                logger.error("Heartbeat error: %s", exc)
            
            self._stop_event.wait(timeout=self._config.heartbeat_interval_sec)
        
        logger.debug("Heartbeat loop ended")
    
    def _send_heartbeat(self) -> None:
        """Send a heartbeat to the server."""
        sample_count = len(self._edge_client.training_buffer)
        is_training = self._state == FederatedClientState.TRAINING
        
        self._transport.send_heartbeat(
            device_id=self.device_id,
            model_version=self._current_model_version,
            sample_count=sample_count,
            is_training=is_training,
        )
    
    # ============================================================
    # Model Polling & Hot-Swap
    # ============================================================
    
    def _poll_for_model(self) -> None:
        """Poll server for new aggregated model."""
        model = self._transport.poll_aggregated_model(self.device_id)
        
        if model is not None and model.version > self._current_model_version:
            self._apply_aggregated_model(model)
    
    def _apply_aggregated_model(self, model) -> None:
        """
        Apply an aggregated model from server.
        
        This triggers:
        1. ONNX hot-swap on EdgeClient
        2. Version update
        """
        logger.info(
            "Applying aggregated model: v%d -> v%d",
            self._current_model_version,
            model.version,
        )
        
        # Hot-swap ONNX if path is available
        if model.onnx_path and os.path.isfile(model.onnx_path):
            try:
                self._edge_client.update_onnx_model(
                    new_model_path=model.onnx_path,
                    new_version=model.version,
                )
                logger.info("ONNX hot-swap complete: v%d", model.version)
            except Exception as exc:
                logger.error("ONNX hot-swap failed: %s", exc)
        
        with self._lock:
            self._current_model_version = model.version
    
    # ============================================================
    # Training Cycle
    # ============================================================
    
    def _training_cycle(self) -> None:
        """
        Perform a training cycle:
        1. Check samples
        2. Train locally
        3. Submit update
        """
        training_buffer = self._edge_client.training_buffer
        sample_count = len(training_buffer)
        
        if sample_count < self._config.min_samples_for_training:
            logger.debug(
                "Not enough samples for training: %d < %d",
                sample_count,
                self._config.min_samples_for_training,
            )
            with self._lock:
                self._state = FederatedClientState.COLLECTING
            return
        
        # Get current global weights to initialize training
        current_model = self._transport.get_current_model()
        initial_weights = current_model.state_dict
        
        # Train locally
        with self._lock:
            self._state = FederatedClientState.TRAINING
        
        logger.info("Starting local training with %d samples", sample_count)
        
        result = self._trainer.train(
            training_buffer=training_buffer,
            initial_state_dict=initial_weights,
            max_epochs=self._config.max_local_epochs,
            batch_size=self._config.batch_size,
            min_samples=self._config.min_samples_for_training,
        )
        
        if not result.success:
            logger.warning("Training failed: %s", result.error_message)
            with self._lock:
                self._state = FederatedClientState.COLLECTING
            return
        
        # Submit update to server
        with self._lock:
            self._state = FederatedClientState.SUBMITTING
        
        ack = self._transport.submit_update(
            device_id=self.device_id,
            state_dict=result.state_dict,
            num_samples=result.samples_used,
            base_version=self._current_model_version,
        )
        
        if ack.success:
            with self._lock:
                self._training_rounds += 1
                self._samples_trained += result.samples_used
                self._last_training_time = time.time()
            
            logger.info(
                "Training round %d complete: %d samples, loss=%.4f",
                self._training_rounds,
                result.samples_used,
                result.final_loss,
            )
            
            # Clear training buffer after successful submission
            training_buffer.clear()
        else:
            logger.warning("Update submission failed: %s", ack.error_message)
        
        with self._lock:
            self._state = FederatedClientState.COLLECTING
    
    # ============================================================
    # Manual Controls (for testing)
    # ============================================================
    
    def force_training(self) -> TrainingResult:
        """
        Force a training cycle immediately.
        
        Useful for testing.
        
        Returns:
            TrainingResult from training.
        """
        training_buffer = self._edge_client.training_buffer
        current_model = self._transport.get_current_model()
        
        return self._trainer.train(
            training_buffer=training_buffer,
            initial_state_dict=current_model.state_dict,
            max_epochs=self._config.max_local_epochs,
            batch_size=self._config.batch_size,
            min_samples=1,  # Allow any number of samples for testing
        )
    
    def force_submit(self, state_dict, num_samples: int) -> bool:
        """
        Force submit an update.
        
        Useful for testing.
        
        Args:
            state_dict: Model weights to submit.
            num_samples: Number of samples used.
        
        Returns:
            True if submission succeeded.
        """
        ack = self._transport.submit_update(
            device_id=self.device_id,
            state_dict=state_dict,
            num_samples=num_samples,
            base_version=self._current_model_version,
        )
        return ack.success
    
    # ============================================================
    # Statistics
    # ============================================================
    
    def get_stats(self) -> dict:
        """Get client statistics."""
        with self._lock:
            return {
                "device_id": self.device_id,
                "state": self._state.value,
                "model_version": self._current_model_version,
                "is_registered": self._is_registered,
                "training_rounds": self._training_rounds,
                "samples_trained": self._samples_trained,
                "samples_buffered": len(self._edge_client.training_buffer),
                "last_training_time": self._last_training_time,
            }
