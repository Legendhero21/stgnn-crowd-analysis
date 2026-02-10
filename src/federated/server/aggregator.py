"""
Aggregator (FedAvg)
-------------------
Federated Averaging implementation for STGNN weight aggregation.

Responsibilities:
- Accept multiple client updates (state_dict, num_samples, base_version)
- Validate version compatibility
- Perform weighted averaging: w_global = Σ (n_i / Σn) * w_i
- Return aggregated state_dict
- Stateless except during a single aggregation round
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch


# Type alias
StateDict = Dict[str, Any]


logger = logging.getLogger(__name__)


@dataclass
class ClientUpdate:
    """
    A single client update for aggregation.
    
    Attributes:
        device_id: Identifier of the submitting device.
        state_dict: PyTorch model weights.
        num_samples: Number of training samples used.
        base_version: Model version this update is based on.
    """
    device_id: str
    state_dict: StateDict
    num_samples: int
    base_version: int
    
    def __post_init__(self):
        if self.num_samples < 0:
            raise ValueError("num_samples must be >= 0")


@dataclass
class AggregationResult:
    """
    Result of a FedAvg aggregation round.
    
    Attributes:
        aggregated_state_dict: Weighted average of all client weights.
        total_samples: Total samples across all clients.
        num_clients: Number of clients that participated.
        participating_devices: List of device IDs.
        base_version: Common base version.
        success: Whether aggregation succeeded.
        error_message: Error message if failed.
    """
    aggregated_state_dict: Optional[StateDict] = None
    total_samples: int = 0
    num_clients: int = 0
    participating_devices: List[str] = field(default_factory=list)
    base_version: int = 0
    success: bool = True
    error_message: str = ""


class Aggregator:
    """
    Federated Averaging (FedAvg) aggregator.
    
    Implements weighted averaging of model weights:
        w_global = Σ (n_i / Σn) * w_i
    
    Where:
        - n_i = number of samples from client i
        - Σn = total samples across all clients
        - w_i = weights from client i
    
    Usage:
        aggregator = Aggregator()
        
        # Collect updates during a round
        aggregator.add_update(device_id, state_dict, num_samples, base_version)
        aggregator.add_update(...)
        
        # Perform aggregation
        result = aggregator.aggregate(expected_version=1)
        
        # Clear for next round
        aggregator.clear()
    """
    
    def __init__(self, expected_param_keys: Optional[Set[str]] = None):
        """
        Initialize aggregator.
        
        Args:
            expected_param_keys: Optional set of expected parameter keys
                                 for validation. If None, inferred from
                                 first update.
        """
        self._expected_keys = expected_param_keys
        self._updates: List[ClientUpdate] = []
    
    def add_update(
        self,
        device_id: str,
        state_dict: StateDict,
        num_samples: int,
        base_version: int,
    ) -> bool:
        """
        Add a client update to the current round.
        
        Args:
            device_id: Identifier of submitting device.
            state_dict: Model weights from client.
            num_samples: Number of training samples used.
            base_version: Model version this update is based on.
        
        Returns:
            True if update was accepted, False otherwise.
        """
        if state_dict is None or len(state_dict) == 0:
            logger.warning("Rejected update from %s: empty state_dict", device_id)
            return False
        
        if num_samples <= 0:
            logger.warning("Rejected update from %s: no samples", device_id)
            return False
        
        # Validate keys
        incoming_keys = set(state_dict.keys())
        
        if self._expected_keys is None:
            # Infer from first update
            self._expected_keys = incoming_keys
        elif incoming_keys != self._expected_keys:
            logger.warning(
                "Rejected update from %s: key mismatch (got %d, expected %d)",
                device_id, len(incoming_keys), len(self._expected_keys),
            )
            return False
        
        # Check for duplicate device
        for existing in self._updates:
            if existing.device_id == device_id:
                logger.warning("Duplicate update from %s, replacing", device_id)
                self._updates.remove(existing)
                break
        
        update = ClientUpdate(
            device_id=device_id,
            state_dict=state_dict,
            num_samples=num_samples,
            base_version=base_version,
        )
        
        self._updates.append(update)
        
        logger.debug(
            "Added update from %s: %d samples, base_version=%d",
            device_id, num_samples, base_version,
        )
        
        return True
    
    @property
    def update_count(self) -> int:
        """Number of updates in current round."""
        return len(self._updates)
    
    @property
    def total_samples(self) -> int:
        """Total samples across all updates."""
        return sum(u.num_samples for u in self._updates)
    
    def get_device_ids(self) -> List[str]:
        """Get list of devices that have submitted updates."""
        return [u.device_id for u in self._updates]
    
    def clear(self) -> None:
        """Clear all updates for next round."""
        self._updates.clear()
        logger.debug("Aggregator cleared")
    
    def aggregate(
        self,
        expected_version: Optional[int] = None,
        min_clients: int = 1,
    ) -> AggregationResult:
        """
        Perform FedAvg aggregation.
        
        Args:
            expected_version: If set, all updates must have this base_version.
            min_clients: Minimum number of clients required.
        
        Returns:
            AggregationResult with aggregated weights or error.
        """
        # Validate minimum clients
        if len(self._updates) < min_clients:
            return AggregationResult(
                success=False,
                error_message=f"Not enough clients: {len(self._updates)} < {min_clients}",
                num_clients=len(self._updates),
            )
        
        # Validate version compatibility
        if expected_version is not None:
            for update in self._updates:
                if update.base_version != expected_version:
                    return AggregationResult(
                        success=False,
                        error_message=f"Version mismatch: {update.device_id} has "
                                      f"base_version={update.base_version}, "
                                      f"expected={expected_version}",
                    )
        
        # Collect base version (use first if not specified)
        base_version = expected_version or self._updates[0].base_version
        
        # Calculate total samples
        total_samples = sum(u.num_samples for u in self._updates)
        
        if total_samples == 0:
            return AggregationResult(
                success=False,
                error_message="Total samples is zero",
            )
        
        # Perform weighted averaging
        try:
            aggregated = self._weighted_average(total_samples)
        except Exception as exc:
            logger.error("Aggregation failed: %s", exc)
            return AggregationResult(
                success=False,
                error_message=str(exc),
            )
        
        device_ids = [u.device_id for u in self._updates]
        
        logger.info(
            "FedAvg complete: %d clients, %d total samples",
            len(self._updates), total_samples,
        )
        
        return AggregationResult(
            aggregated_state_dict=aggregated,
            total_samples=total_samples,
            num_clients=len(self._updates),
            participating_devices=device_ids,
            base_version=base_version,
            success=True,
        )
    
    def _weighted_average(self, total_samples: int) -> StateDict:
        """
        Compute weighted average of state dicts.
        
        Formula: w_global = Σ (n_i / Σn) * w_i
        
        Args:
            total_samples: Sum of all sample counts.
        
        Returns:
            Aggregated state_dict.
        """
        if len(self._updates) == 0:
            raise ValueError("No updates to aggregate")
        
        if total_samples == 0:
            raise ValueError("Total samples is zero")
        
        # Initialize with zeros
        first_state = self._updates[0].state_dict
        aggregated: StateDict = {}
        
        for key in first_state.keys():
            aggregated[key] = torch.zeros_like(
                first_state[key], dtype=torch.float32
            )
        
        # Weighted sum
        for update in self._updates:
            weight = update.num_samples / total_samples
            
            for key in aggregated.keys():
                aggregated[key] += weight * update.state_dict[key].float()
        
        # Restore original dtypes
        for key in aggregated.keys():
            original_dtype = first_state[key].dtype
            aggregated[key] = aggregated[key].to(original_dtype)
        
        return aggregated


# ============================================================
# Test Utilities
# ============================================================

def verify_fedavg_math(
    updates: List[Tuple[StateDict, int]],
) -> Tuple[StateDict, bool]:
    """
    Verify FedAvg produces correct weighted average.
    
    Args:
        updates: List of (state_dict, num_samples) tuples.
    
    Returns:
        Tuple of (aggregated_state_dict, is_correct).
    """
    if not updates:
        return {}, False
    
    aggregator = Aggregator()
    
    for i, (state_dict, num_samples) in enumerate(updates):
        aggregator.add_update(
            device_id=f"test-device-{i}",
            state_dict=state_dict,
            num_samples=num_samples,
            base_version=0,
        )
    
    result = aggregator.aggregate()
    
    if not result.success:
        return {}, False
    
    # Verify by manual computation
    total = sum(n for _, n in updates)
    expected: StateDict = {}
    
    for key in updates[0][0].keys():
        expected[key] = torch.zeros_like(updates[0][0][key], dtype=torch.float32)
        
        for state_dict, num_samples in updates:
            weight = num_samples / total
            expected[key] += weight * state_dict[key].float()
        
        expected[key] = expected[key].to(updates[0][0][key].dtype)
    
    # Compare
    for key in expected.keys():
        if not torch.allclose(
            result.aggregated_state_dict[key].float(),
            expected[key].float(),
            atol=1e-6,
        ):
            return result.aggregated_state_dict, False
    
    return result.aggregated_state_dict, True
