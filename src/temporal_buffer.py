import numpy as np
from typing import Optional, Dict, List


class TemporalGraphBuffer:
    """
    Maintains a fixed-length window of consecutive graph snapshots and
    returns a temporal tensor once the buffer is full.

    Each graph is expected to be a dict with at least key "x" -> np.ndarray
    of shape (N, F). Node count is assumed stable over the window; if it
    changes, the buffer is reset.
    """

    def __init__(self, window_size: int = 5):
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        self.window_size: int = int(window_size)
        self.buffer: List[Dict[str, np.ndarray]] = []
        self._node_count: Optional[int] = None

    def reset(self) -> None:
        """Clear buffer and node-count tracking."""
        self.buffer.clear()
        self._node_count = None

    def push(self, graph: Optional[Dict[str, np.ndarray]]) -> Optional[np.ndarray]:
        """
        Add a new graph to buffer and return temporal tensor if ready.

        Args:
            graph: Dict with key "x" (node features [N, F]) or None.

        Returns:
            x_seq: np.ndarray with shape (1, T, N, F) when buffer is full,
                   otherwise None.
        """
        # Missing graph ⇒ break temporal continuity.
        if graph is None:
            self.reset()
            return None

        if "x" not in graph:
            raise KeyError("Graph dict must contain key 'x' with node features.")

        x = graph["x"]
        if not isinstance(x, np.ndarray):
            raise TypeError(f"graph['x'] must be a np.ndarray, got {type(x)}")

        if x.ndim != 2:
            raise ValueError(f"graph['x'] must be 2D [N, F], got shape {x.shape}")

        n_nodes = x.shape[0]

        # Initialize node count on first valid graph
        if self._node_count is None:
            self._node_count = n_nodes
        # Node count changed ⇒ reset buffer (track new crowd configuration)
        elif n_nodes != self._node_count:
            self.reset()
            self._node_count = n_nodes

        self.buffer.append(graph)

        # Trim to window size
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        # Not enough temporal context yet
        if len(self.buffer) < self.window_size:
            return None

        # Stack [T, N, F] and add batch dimension → [1, T, N, F]
        try:
            x_seq = np.stack([g["x"] for g in self.buffer], axis=0)
        except Exception as exc:
            # If shapes don’t align, reset to avoid silent misuse
            self.reset()
            raise ValueError(f"Failed to stack graphs into temporal tensor: {exc}")

        x_seq = x_seq[np.newaxis, ...].astype(np.float32, copy=False)
        return x_seq
