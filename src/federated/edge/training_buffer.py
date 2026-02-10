"""
Training Buffer
---------------
Collects training samples from TemporalBuffer output for local federated training.

Design:
- Collects AFTER TemporalBuffer produces complete [1, T, N, F] sequences
- Stores (x_seq, edge_index, target) tuples
- Fixed capacity with FIFO eviction
- Thread-safe for concurrent read/write

The target is the actual next-frame positions, used for computing prediction loss.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterator, List, Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class TrainingSample:
    """
    A single training sample for STGNN local training.
    
    Attributes:
        x_seq: Temporal sequence [1, T, N, 5] from TemporalBuffer.
        edge_index: Graph edges [2, E] from the last frame's graph.
        target: Ground truth next-frame positions [N, 2].
        timestamp: Unix timestamp when sample was collected.
        frame_idx: Frame index when sample was created.
    """
    x_seq: np.ndarray          # [1, T, N, 5]
    edge_index: np.ndarray     # [2, E]
    target: np.ndarray         # [N, 2]
    timestamp: float = field(default_factory=time.time)
    frame_idx: int = 0
    
    def __post_init__(self):
        """Validate sample dimensions."""
        if self.x_seq.ndim != 4:
            raise ValueError(f"x_seq must be 4D [1, T, N, F], got shape {self.x_seq.shape}")
        
        if self.x_seq.shape[0] != 1:
            raise ValueError(f"x_seq batch must be 1, got {self.x_seq.shape[0]}")
        
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must be [2, E], got shape {self.edge_index.shape}")
        
        if self.target.ndim != 2 or self.target.shape[1] != 2:
            raise ValueError(f"target must be [N, 2], got shape {self.target.shape}")
        
        n_nodes = self.x_seq.shape[2]
        if self.target.shape[0] != n_nodes:
            raise ValueError(
                f"target node count ({self.target.shape[0]}) doesn't match "
                f"x_seq node count ({n_nodes})"
            )
    
    @property
    def n_nodes(self) -> int:
        """Number of nodes in this sample."""
        return self.x_seq.shape[2]
    
    @property
    def n_edges(self) -> int:
        """Number of edges in this sample."""
        return self.edge_index.shape[1]
    
    @property
    def temporal_length(self) -> int:
        """Temporal window length (T)."""
        return self.x_seq.shape[1]
    
    def to_tensors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return sample components as a tuple.
        
        Returns:
            (x_seq, edge_index, target) tuple.
        """
        return self.x_seq, self.edge_index, self.target
    
    def copy(self) -> "TrainingSample":
        """Create a deep copy of this sample."""
        return TrainingSample(
            x_seq=self.x_seq.copy(),
            edge_index=self.edge_index.copy(),
            target=self.target.copy(),
            timestamp=self.timestamp,
            frame_idx=self.frame_idx,
        )


class TrainingBuffer:
    """
    Thread-safe buffer for collecting STGNN training samples.
    
    Samples are collected AFTER TemporalBuffer produces complete sequences.
    The buffer has a fixed maximum capacity; oldest samples are evicted
    when full (FIFO).
    
    Usage:
        buffer = TrainingBuffer(max_samples=1000)
        
        # In pipeline loop (after TemporalBuffer)
        if x_seq is not None and next_positions is not None:
            buffer.add(x_seq, edge_index, next_positions, frame_idx)
        
        # In training loop
        for batch in buffer.iter_batches(batch_size=32):
            train_on_batch(batch)
    """
    
    def __init__(
        self,
        max_samples: int = 1000,
        min_samples_for_training: int = 50,
    ):
        """
        Initialize training buffer.
        
        Args:
            max_samples: Maximum number of samples to store.
            min_samples_for_training: Minimum samples required before training.
        
        Raises:
            ValueError: If max_samples < min_samples_for_training.
        """
        if max_samples < min_samples_for_training:
            raise ValueError(
                f"max_samples ({max_samples}) must be >= "
                f"min_samples_for_training ({min_samples_for_training})"
            )
        
        self._max_samples = max_samples
        self._min_samples = min_samples_for_training
        
        self._buffer: Deque[TrainingSample] = deque(maxlen=max_samples)
        self._lock = threading.RLock()
        
        # Statistics
        self._total_added = 0
        self._total_evicted = 0
        self._last_add_time: Optional[float] = None
    
    def add(
        self,
        x_seq: np.ndarray,
        edge_index: np.ndarray,
        next_frame_positions: np.ndarray,
        frame_idx: int = 0,
    ) -> bool:
        """
        Add a training sample to the buffer.
        
        This should be called AFTER TemporalBuffer.push() returns a valid
        x_seq, and when the NEXT frame's positions are available.
        
        Args:
            x_seq: Temporal sequence [1, T, N, 5] from TemporalBuffer.
            edge_index: Graph edges [2, E] from current graph.
            next_frame_positions: Actual positions in next frame [N, 2].
            frame_idx: Frame index for tracking.
        
        Returns:
            True if sample was added successfully.
        
        Note:
            Invalid samples are silently rejected (logged as debug).
        """
        # Validate inputs
        if x_seq is None or x_seq.size == 0:
            logger.debug("Rejected sample: x_seq is empty")
            return False
        
        if edge_index is None or edge_index.size == 0:
            logger.debug("Rejected sample: edge_index is empty")
            return False
        
        if next_frame_positions is None or next_frame_positions.size == 0:
            logger.debug("Rejected sample: next_frame_positions is empty")
            return False
        
        # Ensure correct shapes
        if x_seq.ndim != 4 or x_seq.shape[0] != 1:
            logger.debug("Rejected sample: x_seq shape %s invalid", x_seq.shape)
            return False
        
        n_nodes = x_seq.shape[2]
        
        if next_frame_positions.shape[0] != n_nodes:
            logger.debug(
                "Rejected sample: node count mismatch (x_seq=%d, target=%d)",
                n_nodes,
                next_frame_positions.shape[0],
            )
            return False
        
        # Ensure we only take x,y from positions
        if next_frame_positions.ndim == 2 and next_frame_positions.shape[1] >= 2:
            target = next_frame_positions[:, :2].copy()
        else:
            logger.debug("Rejected sample: invalid target shape %s", next_frame_positions.shape)
            return False
        
        # Create sample
        try:
            sample = TrainingSample(
                x_seq=x_seq.copy().astype(np.float32),
                edge_index=edge_index.copy().astype(np.int64),
                target=target.astype(np.float32),
                timestamp=time.time(),
                frame_idx=frame_idx,
            )
        except Exception as exc:
            logger.debug("Failed to create sample: %s", exc)
            return False
        
        # Add to buffer
        with self._lock:
            was_full = len(self._buffer) == self._max_samples
            self._buffer.append(sample)
            self._total_added += 1
            if was_full:
                self._total_evicted += 1
            self._last_add_time = sample.timestamp
        
        return True
    
    def add_from_graphs(
        self,
        current_x_seq: np.ndarray,
        current_edge_index: np.ndarray,
        next_graph: dict,
        frame_idx: int = 0,
    ) -> bool:
        """
        Add sample using current sequence and next frame's graph.
        
        This is a convenience method that extracts positions from the
        next frame's graph dict.
        
        Args:
            current_x_seq: Current temporal sequence [1, T, N, 5].
            current_edge_index: Current graph edges [2, E].
            next_graph: Next frame's graph dict with 'x' key.
            frame_idx: Frame index.
        
        Returns:
            True if sample was added.
        """
        if next_graph is None or "x" not in next_graph:
            return False
        
        next_features = next_graph["x"]
        if next_features is None or next_features.shape[0] == 0:
            return False
        
        # Extract positions (first 2 columns: x, y)
        next_positions = next_features[:, :2]
        
        return self.add(current_x_seq, current_edge_index, next_positions, frame_idx)
    
    def __len__(self) -> int:
        """Number of samples in buffer."""
        with self._lock:
            return len(self._buffer)
    
    def is_ready_for_training(self) -> bool:
        """Check if buffer has enough samples for training."""
        with self._lock:
            return len(self._buffer) >= self._min_samples
    
    def clear(self) -> int:
        """
        Clear all samples from buffer.
        
        Returns:
            Number of samples cleared.
        """
        with self._lock:
            count = len(self._buffer)
            self._buffer.clear()
            return count
    
    def get_all(self) -> List[TrainingSample]:
        """
        Get all samples as a list.
        
        Returns:
            List of all samples (copies).
        """
        with self._lock:
            return [sample.copy() for sample in self._buffer]
    
    def get_recent(self, n: int) -> List[TrainingSample]:
        """
        Get the N most recent samples.
        
        Args:
            n: Number of samples to get.
        
        Returns:
            List of most recent samples (copies).
        """
        with self._lock:
            samples = list(self._buffer)[-n:]
            return [s.copy() for s in samples]
    
    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> Iterator[List[TrainingSample]]:
        """
        Iterate over samples in batches.
        
        Thread-safe: creates a snapshot of samples at iteration start.
        
        Args:
            batch_size: Number of samples per batch.
            shuffle: If True, shuffle samples before batching.
            drop_last: If True, drop incomplete final batch.
        
        Yields:
            Lists of TrainingSample of size batch_size (or less for last batch).
        """
        with self._lock:
            samples = list(self._buffer)
        
        if shuffle:
            np.random.shuffle(samples)
        
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            
            if drop_last and len(batch) < batch_size:
                break
            
            yield batch
    
    def to_numpy_batch(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Convert all samples to stacked numpy arrays.
        
        Returns:
            Tuple of (x_seq_batch, edge_indices, targets) or None if empty.
            
            - x_seq_batch: [N_samples, T, N_nodes, F]
            - edge_indices: List of [2, E_i] arrays (variable edges per sample)
            - targets: [N_samples, N_nodes, 2]
        
        Note:
            This only works if all samples have the same node count.
            Variable node counts require iter_batches instead.
        """
        with self._lock:
            if len(self._buffer) == 0:
                return None
            
            samples = list(self._buffer)
        
        # Check for consistent node counts
        node_counts = set(s.n_nodes for s in samples)
        if len(node_counts) > 1:
            logger.warning(
                "Cannot create batch: variable node counts %s",
                node_counts,
            )
            return None
        
        try:
            # Stack x_seq: [N, 1, T, N_nodes, F] -> [N, T, N_nodes, F]
            x_seqs = np.concatenate([s.x_seq for s in samples], axis=0)
            
            # Collect edge indices (variable length)
            edge_indices = [s.edge_index for s in samples]
            
            # Stack targets: [N, N_nodes, 2]
            targets = np.stack([s.target for s in samples], axis=0)
            
            return x_seqs, edge_indices, targets
            
        except Exception as exc:
            logger.error("Failed to create numpy batch: %s", exc)
            return None
    
    def get_stats(self) -> dict:
        """
        Get buffer statistics.
        
        Returns:
            Dict with count, capacity, added, evicted, etc.
        """
        with self._lock:
            current_count = len(self._buffer)
            
            if current_count > 0:
                avg_nodes = np.mean([s.n_nodes for s in self._buffer])
                avg_edges = np.mean([s.n_edges for s in self._buffer])
            else:
                avg_nodes = 0
                avg_edges = 0
            
            return {
                "count": current_count,
                "capacity": self._max_samples,
                "fill_ratio": current_count / self._max_samples,
                "total_added": self._total_added,
                "total_evicted": self._total_evicted,
                "avg_nodes": avg_nodes,
                "avg_edges": avg_edges,
                "ready_for_training": current_count >= self._min_samples,
                "last_add_time": self._last_add_time,
            }
