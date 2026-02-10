"""
Device Registry
---------------
Tracks edge devices in the federated learning system.

Responsibilities:
- Register edge devices on join
- Track device state (device_id, last_seen, model_version, sample_count)
- Mark devices as ACTIVE / STALE based on timeout
- Support device enumeration and filtering
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


logger = logging.getLogger(__name__)


class DeviceStatus(Enum):
    """Status of an edge device."""
    ACTIVE = "ACTIVE"       # Recently seen
    STALE = "STALE"         # Not seen within timeout
    DISCONNECTED = "DISCONNECTED"  # Explicitly disconnected


@dataclass
class DeviceInfo:
    """
    Information about a registered edge device.
    
    Attributes:
        device_id: Unique identifier.
        device_type: Category (drone, raspi, laptop, etc.).
        registered_at: Unix timestamp of registration.
        last_seen: Unix timestamp of last activity.
        current_model_version: Model version the device has.
        last_sample_count: Sample count from last heartbeat.
        last_update_samples: Samples used in last update submission.
        status: Current device status.
    """
    device_id: str
    device_type: str = "laptop"
    registered_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    current_model_version: int = 0
    last_sample_count: int = 0
    last_update_samples: int = 0
    status: DeviceStatus = DeviceStatus.ACTIVE
    
    def update_seen(self) -> None:
        """Update last_seen timestamp."""
        self.last_seen = time.time()
        if self.status == DeviceStatus.STALE:
            self.status = DeviceStatus.ACTIVE
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "device_id": self.device_id,
            "device_type": self.device_type,
            "registered_at": self.registered_at,
            "last_seen": self.last_seen,
            "current_model_version": self.current_model_version,
            "last_sample_count": self.last_sample_count,
            "status": self.status.value,
        }


class DeviceRegistry:
    """
    Registry for tracking edge devices.
    
    Thread-safe operations for device registration and status tracking.
    
    Usage:
        registry = DeviceRegistry(stale_timeout_sec=60)
        
        # Register device
        registry.register("device-001", "drone", model_version=0)
        
        # Update on activity
        registry.update_heartbeat("device-001", model_version=1, sample_count=100)
        
        # Get active devices
        active = registry.get_active_devices()
        
        # Mark stale devices
        registry.mark_stale_devices()
    """
    
    def __init__(self, stale_timeout_sec: float = 120.0):
        """
        Initialize device registry.
        
        Args:
            stale_timeout_sec: Seconds without activity before device is marked stale.
        """
        self._stale_timeout = stale_timeout_sec
        self._devices: Dict[str, DeviceInfo] = {}
        self._lock = threading.RLock()
    
    def register(
        self,
        device_id: str,
        device_type: str = "laptop",
        model_version: int = 0,
    ) -> DeviceInfo:
        """
        Register a new device or update existing registration.
        
        Args:
            device_id: Unique device identifier.
            device_type: Category of device.
            model_version: Current model version on device.
        
        Returns:
            DeviceInfo for the registered device.
        """
        with self._lock:
            if device_id in self._devices:
                # Update existing
                device = self._devices[device_id]
                device.update_seen()
                device.current_model_version = model_version
                device.status = DeviceStatus.ACTIVE
                logger.info("Re-registered device: %s", device_id)
            else:
                # New registration
                device = DeviceInfo(
                    device_id=device_id,
                    device_type=device_type,
                    current_model_version=model_version,
                    status=DeviceStatus.ACTIVE,
                )
                self._devices[device_id] = device
                logger.info("Registered new device: %s (%s)", device_id, device_type)
            
            return device
    
    def unregister(self, device_id: str) -> bool:
        """
        Unregister a device.
        
        Args:
            device_id: Device to remove.
        
        Returns:
            True if device was removed, False if not found.
        """
        with self._lock:
            if device_id in self._devices:
                del self._devices[device_id]
                logger.info("Unregistered device: %s", device_id)
                return True
            return False
    
    def get(self, device_id: str) -> Optional[DeviceInfo]:
        """
        Get device info by ID.
        
        Args:
            device_id: Device to look up.
        
        Returns:
            DeviceInfo or None if not found.
        """
        with self._lock:
            return self._devices.get(device_id)
    
    def update_heartbeat(
        self,
        device_id: str,
        model_version: Optional[int] = None,
        sample_count: Optional[int] = None,
    ) -> bool:
        """
        Update device on heartbeat.
        
        Args:
            device_id: Device sending heartbeat.
            model_version: Current model version (if provided).
            sample_count: Current sample count (if provided).
        
        Returns:
            True if device exists and was updated.
        """
        with self._lock:
            device = self._devices.get(device_id)
            if device is None:
                logger.warning("Heartbeat from unknown device: %s", device_id)
                return False
            
            device.update_seen()
            
            if model_version is not None:
                device.current_model_version = model_version
            
            if sample_count is not None:
                device.last_sample_count = sample_count
            
            return True
    
    def record_update_submission(
        self,
        device_id: str,
        sample_count: int,
        model_version: int,
    ) -> bool:
        """
        Record that a device submitted a model update.
        
        Args:
            device_id: Device that submitted.
            sample_count: Samples in the update.
            model_version: Version the update was based on.
        
        Returns:
            True if device exists and was updated.
        """
        with self._lock:
            device = self._devices.get(device_id)
            if device is None:
                return False
            
            device.update_seen()
            device.last_update_samples = sample_count
            device.current_model_version = model_version
            
            return True
    
    def update_model_version(
        self,
        device_ids: List[str],
        new_version: int,
    ) -> int:
        """
        Update model version for multiple devices.
        
        Called after distributing aggregated model.
        
        Args:
            device_ids: Devices to update.
            new_version: New model version.
        
        Returns:
            Number of devices updated.
        """
        count = 0
        with self._lock:
            for device_id in device_ids:
                device = self._devices.get(device_id)
                if device is not None:
                    device.current_model_version = new_version
                    count += 1
        return count
    
    def mark_stale_devices(self) -> List[str]:
        """
        Mark devices as stale if they haven't been seen recently.
        
        Returns:
            List of device IDs that were marked stale.
        """
        now = time.time()
        stale_ids = []
        
        with self._lock:
            for device_id, device in self._devices.items():
                if device.status == DeviceStatus.ACTIVE:
                    if now - device.last_seen > self._stale_timeout:
                        device.status = DeviceStatus.STALE
                        stale_ids.append(device_id)
                        logger.info("Marked device as stale: %s", device_id)
        
        return stale_ids
    
    def get_active_devices(self) -> List[DeviceInfo]:
        """Get all active devices."""
        with self._lock:
            return [
                d for d in self._devices.values()
                if d.status == DeviceStatus.ACTIVE
            ]
    
    def get_stale_devices(self) -> List[DeviceInfo]:
        """Get all stale devices."""
        with self._lock:
            return [
                d for d in self._devices.values()
                if d.status == DeviceStatus.STALE
            ]
    
    def get_all_devices(self) -> List[DeviceInfo]:
        """Get all registered devices."""
        with self._lock:
            return list(self._devices.values())
    
    def get_device_ids(self) -> Set[str]:
        """Get set of all device IDs."""
        with self._lock:
            return set(self._devices.keys())
    
    @property
    def count(self) -> int:
        """Total number of registered devices."""
        with self._lock:
            return len(self._devices)
    
    @property
    def active_count(self) -> int:
        """Number of active devices."""
        with self._lock:
            return sum(
                1 for d in self._devices.values()
                if d.status == DeviceStatus.ACTIVE
            )
    
    def get_stats(self) -> dict:
        """Get registry statistics."""
        with self._lock:
            active = sum(1 for d in self._devices.values() if d.status == DeviceStatus.ACTIVE)
            stale = sum(1 for d in self._devices.values() if d.status == DeviceStatus.STALE)
            
            return {
                "total_devices": len(self._devices),
                "active_devices": active,
                "stale_devices": stale,
                "stale_timeout_sec": self._stale_timeout,
            }
    
    def clear(self) -> int:
        """
        Clear all registered devices.
        
        Returns:
            Number of devices cleared.
        """
        with self._lock:
            count = len(self._devices)
            self._devices.clear()
            logger.info("Cleared %d devices from registry", count)
            return count
