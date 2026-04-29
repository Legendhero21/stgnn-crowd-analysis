"""
Local Transport Adapter
-----------------------
In-process transport for federated learning communication.

This adapter calls FederatedServer methods directly, making it
ideal for testing and single-machine simulations. The interface
is designed to be replaceable with a network transport (sockets, gRPC)
without changing the FederatedClient code.

Usage:
    server = FederatedServer(config)
    transport = LocalTransport(server)
    
    # Now use transport for all server communication
    ack = transport.register_device(device_id, device_type)
    model = transport.poll_aggregated_model(device_id)
"""

from __future__ import annotations

import logging
import os
import torch
from pathlib import Path

SHARED_DIR = r"\\VEDANTG\BE_Project_edge"
UPDATES_DIR = os.path.join(SHARED_DIR, "updates")

os.makedirs(UPDATES_DIR, exist_ok=True)

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from ..protocol.messages import (
    RegisterDevice,
    SubmitUpdate,
    Heartbeat,
    RegisterAck,
    AggregatedModel,
    UpdateAck,
    StateDict,
    create_heartbeat,
)

logger = logging.getLogger(__name__)


# ============================================================
# Transport Protocol (Interface)
# ============================================================

@runtime_checkable
class TransportProtocol(Protocol):
    """
    Protocol defining the transport interface.
    
    Any transport implementation must provide these methods.
    This allows FederatedClient to work with any transport type.
    """
    
    def register_device(
        self,
        device_id: str,
        device_type: str = "laptop",
        current_model_version: int = 0,
    ) -> RegisterAck:
        """Register device with server."""
        ...
    
    def submit_update(
        self,
        device_id: str,
        state_dict: StateDict,
        num_samples: int,
        base_version: int,
    ) -> UpdateAck:
        """Submit model update to server."""
        ...
    
    def poll_aggregated_model(
        self,
        device_id: str,
    ) -> Optional[AggregatedModel]:
        """Poll for new aggregated model."""
        ...
    
    def send_heartbeat(
        self,
        device_id: str,
        model_version: int,
        sample_count: int = 0,
        is_training: bool = False,
    ) -> bool:
        """Send heartbeat to server."""
        ...
    
    def get_current_model(self) -> AggregatedModel:
        """Get the current global model (for initial sync)."""
        ...


# ============================================================
# Local Transport Implementation
# ============================================================

class LocalTransport:
    """
    In-process transport that calls FederatedServer directly.
    
    This is the transport used for local simulation and testing.
    All method calls are synchronous and happen in the same process.
    
    Attributes:
        server: Reference to the FederatedServer instance.
    
    Thread Safety:
        Thread-safe because FederatedServer is thread-safe.
        Multiple FederatedClients can share the same LocalTransport.
    
    Usage:
        server = FederatedServer(config)
        transport = LocalTransport(server)
        
        # Register
        ack = transport.register_device("edge-001", "laptop")
        if ack.success:
            print(f"Registered, global version: {ack.current_global_version}")
    """
    
    def __init__(self, server: Any):
        """
        Initialize local transport.
        
        Args:
            server: FederatedServer instance to communicate with.
        """
        # Import here to avoid circular imports
        from ..server.server import FederatedServer
        
        if not isinstance(server, FederatedServer):
            raise TypeError("server must be a FederatedServer instance")
        
        self._server = server
        logger.debug("LocalTransport initialized with server")
    
    def register_device(
        self,
        device_id: str,
        device_type: str = "laptop",
        current_model_version: int = 0,
    ) -> RegisterAck:
        """
        Register device with server.
        
        Args:
            device_id: Unique device identifier.
            device_type: Device type (laptop, edge, etc.).
            current_model_version: Device's current model version.
        
        Returns:
            RegisterAck with registration result.
        """
        msg = RegisterDevice(
            device_id=device_id,
            device_type=device_type,
            current_model_version=current_model_version,
        )
        
        ack = self._server.register_device(msg)
        
        if ack.success:
            logger.info(
                "Device %s registered, global version: %d",
                device_id, ack.current_global_version,
            )
        else:
            logger.error(
                "Device %s registration failed: %s",
                device_id, ack.error_message,
            )
        
        return ack
    
    def submit_update(
        self,
        device_id: str,
        state_dict: StateDict,
        num_samples: int,
        base_version: int,
    ) -> UpdateAck:
        
        msg = SubmitUpdate(
            device_id=device_id,
            state_dict=state_dict,
            num_samples=num_samples,
            base_version=base_version,
        )
        
        """ack = self._server.submit_update(msg)
        
        if ack.success:
            logger.info(
                "Device %s submitted update: %d samples, round %d",
                device_id, num_samples, ack.round_id,
            )
        else:
            logger.error(
                "Device %s update rejected: %s",
                device_id, ack.error_message,
            )
        
        return ack"""
        # -------- FILE-BASED UPDATE --------
        update_path = os.path.join(UPDATES_DIR, f"{device_id}.pt")

        temp_path = update_path + ".tmp"
        torch.save(data, temp_path)
        os.replace(temp_path, update_path)

        logger.info("Device %s wrote update to %s", device_id, update_path)

        # fake ack (since no server call)
        return UpdateAck(success=True, round_id=0, error_message=None)
    
    def poll_aggregated_model(
        self,
        device_id: str,
    ) -> Optional[AggregatedModel]:
        
        """model = self._server.get_aggregated_model(device_id)
        
        if model is not None:
            logger.info(
                "Device %s received aggregated model v%d",
                device_id, model.version,
            )
        
        return model"""
        # -------- FILE-BASED MODEL FETCH --------
        model_path = os.path.join(SHARED_DIR, "global_model.pt")
        version_path = os.path.join(SHARED_DIR, "version.txt")

        if not os.path.exists(model_path) or not os.path.exists(version_path):
            return None

        try:
            version = int(open(version_path).read().strip())
            state_dict = torch.load(model_path, map_location="cpu")

            logger.info("Device %s loaded global model v%d", device_id, version)

            return AggregatedModel(
                version=version,
                state_dict=state_dict
            )
        except Exception as e:
            logger.error("Failed to load global model: %s", str(e))
            return None
    
    def send_heartbeat(
        self,
        device_id: str,
        model_version: int,
        sample_count: int = 0,
        is_training: bool = False,
    ) -> bool:
        """
        Send heartbeat to server.
        
        Args:
            device_id: Device sending heartbeat.
            model_version: Device's current model version.
            sample_count: Number of samples collected.
            is_training: Whether device is currently training.
        
        Returns:
            True if heartbeat was processed.
        """
        msg = create_heartbeat(
            device_id=device_id,
            model_version=model_version,
            sample_count=sample_count,
            is_training=is_training,
        )
        
        return self._server.handle_heartbeat(msg)
    
    def get_current_model(self) -> AggregatedModel:
        """
        Get the current global model.
        
        Used for initial synchronization when a device first connects.
        
        Returns:
            AggregatedModel with current global weights.
        """
        return self._server.get_current_model()
    
    @property
    def server_version(self) -> int:
        """Get current server model version."""
        return self._server.model_version
