"""
Temporal Graph Buffer with Padding & Masking
---------------------------------------------
Maintains a fixed-length window of consecutive graph snapshots.

Key design (Wu et al., 2020; Seo et al., 2018):
- All graphs are padded to MAX_NODES so that the tensor shape is
  constant across frames regardless of how many people are detected.
- A binary mask tracks which nodes are real vs. padded.
- The buffer NEVER resets when node count changes between frames.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple

# Maximum number of nodes per frame. Graphs with fewer nodes are
# zero-padded; graphs with more nodes are truncated (oldest excess dropped).
MAX_NODES: int = 50


class TemporalGraphBuffer:
    """
    Fixed-window temporal buffer that produces [1, T, MAX_NODES, F] tensors.

    Unlike the previous implementation, this buffer does NOT reset when the
    number of detected people changes between frames.  Instead, every graph
    is padded/truncated to MAX_NODES and a binary mask records validity.
    """

    def __init__(self, window_size: int = 5, max_nodes: int = MAX_NODES) -> None:
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if max_nodes <= 0:
            raise ValueError(f"max_nodes must be positive, got {max_nodes}")

        self.window_size: int = int(window_size)
        self.max_nodes: int = int(max_nodes)

        # Each element: {"x_padded": [MAX_NODES, F], "mask": [MAX_NODES]}
        self.buffer: List[Dict[str, np.ndarray]] = []

    def reset(self) -> None:
        """Clear the buffer (e.g. on scene change or explicit request)."""
        self.buffer.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def push(
        self, graph: Optional[Dict[str, np.ndarray]]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Add a new graph snapshot and return the temporal tensor when ready.

        Args:
            graph: Dict with key ``"x"`` holding node features [N, F],
                   or ``None`` when no people are detected.

        Returns:
            (x_seq, mask_seq) where
                x_seq:    np.ndarray  [1, T, MAX_NODES, F]  or None
                mask_seq: np.ndarray  [1, T, MAX_NODES]     or None
        """
        # No detections → skip this frame entirely (do not push).
        if graph is None:
            return None, None

        if "x" not in graph:
            raise KeyError("Graph dict must contain key 'x' with node features.")

        x = graph["x"]
        if not isinstance(x, np.ndarray):
            raise TypeError(f"graph['x'] must be np.ndarray, got {type(x)}")
        if x.ndim != 2:
            raise ValueError(f"graph['x'] must be 2-D [N, F], got shape {x.shape}")

        n_nodes, n_feat = x.shape
        if n_nodes == 0:
            # Zero detections after filtering — treat like None.
            return None, None

        # --- Pad / truncate to MAX_NODES ---
        x_padded = np.zeros((self.max_nodes, n_feat), dtype=np.float32)
        mask = np.zeros(self.max_nodes, dtype=np.float32)

        n_valid = min(n_nodes, self.max_nodes)
        x_padded[:n_valid] = x[:n_valid]
        mask[:n_valid] = 1.0

        self.buffer.append({"x_padded": x_padded, "mask": mask})

        # Trim to window size
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        # Not enough frames yet
        if len(self.buffer) < self.window_size:
            return None, None

        # --- Stack temporal tensor ---
        try:
            x_seq = np.stack(
                [g["x_padded"] for g in self.buffer], axis=0
            )  # [T, MAX_NODES, F]
            mask_seq = np.stack(
                [g["mask"] for g in self.buffer], axis=0
            )  # [T, MAX_NODES]
        except Exception as exc:
            self.reset()
            raise ValueError(f"Failed to stack temporal tensor: {exc}")

        # Add batch dimension → [1, T, MAX_NODES, F] / [1, T, MAX_NODES]
        x_seq = x_seq[np.newaxis, ...].astype(np.float32, copy=False)
        mask_seq = mask_seq[np.newaxis, ...].astype(np.float32, copy=False)

        return x_seq, mask_seq
