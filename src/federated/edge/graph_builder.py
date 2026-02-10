"""
Graph Builder
-------------
Builds spatial graphs from person detections for STGNN.

This is the canonical implementation used by EdgeClient.
Logic mirrors the original RealtimeGraphBuilder from run_pipeline_realtime.py.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist


logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Build spatial graphs from person detections.
    
    Creates node features [x, y, dx, dy, density] and edges based on
    proximity radius in normalized coordinates.
    
    This is a direct adaptation of RealtimeGraphBuilder from run_pipeline_realtime.py.
    """
    
    def __init__(self, radius: float, min_nodes: int = 2) -> None:
        """
        Initialize graph builder.
        
        Args:
            radius: Radius for spatial edges in normalized [0,1] coordinates.
            min_nodes: Minimum nodes required to build a valid graph.
        """
        self.radius = float(radius)
        self.min_nodes = min_nodes
        self.prev_coords: Optional[np.ndarray] = None
    
    def build_graph(
        self,
        detections: List[Tuple[float, float]],
        frame_shape: Tuple[int, int],
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Build a spatial graph from detection centers.
        
        Args:
            detections: List of (x, y) in pixel coordinates.
            frame_shape: (H, W).
        
        Returns:
            Dict with 'x' (node features [N, 5]) and 'edge_index' [2, E],
            or None if too few nodes.
        """
        if not detections or len(detections) < self.min_nodes:
            self.prev_coords = None
            return None
        
        h, w = frame_shape
        if h <= 0 or w <= 0:
            self.prev_coords = None
            return None
        
        coords = np.asarray(detections, dtype=np.float32).copy()
        if coords.ndim != 2 or coords.shape[1] != 2:
            self.prev_coords = None
            return None
        
        # Normalize to [0, 1]
        coords[:, 0] /= float(w)
        coords[:, 1] /= float(h)
        
        # Compute velocity
        if self.prev_coords is not None and len(self.prev_coords) == len(coords):
            velocity = coords - self.prev_coords
        else:
            velocity = np.zeros_like(coords, dtype=np.float32)
        
        # Build edges
        edge_index = self._build_edges(coords)
        
        # Compute density
        density = self._compute_density(len(coords), edge_index)
        
        # Stack features: [x, y, dx, dy, density]
        features = np.hstack([
            coords,
            velocity,
            density[:, None],
        ]).astype(np.float32)
        
        self.prev_coords = coords.copy()
        
        return {"x": features, "edge_index": edge_index}
    
    def _build_edges(self, coords: np.ndarray) -> np.ndarray:
        """Build edges based on radius proximity."""
        if coords.size == 0:
            return np.zeros((2, 0), dtype=np.int64)
        
        dists = cdist(coords, coords)
        row, col = np.where((dists < self.radius) & (dists > 0.0))
        
        if row.size == 0:
            return np.zeros((2, 0), dtype=np.int64)
        
        return np.stack([row, col], axis=0).astype(np.int64)
    
    def _compute_density(self, n: int, edge_index: np.ndarray) -> np.ndarray:
        """Compute normalized node degrees as density."""
        density = np.zeros(n, dtype=np.float32)
        
        if edge_index.shape[1] > 0:
            uniq, cnt = np.unique(edge_index[0], return_counts=True)
            density[uniq] = cnt
        
        if n > 1:
            density /= float(n - 1)
        
        return density
    
    def reset(self) -> None:
        """Reset velocity tracking."""
        self.prev_coords = None
    
    @property
    def current_node_count(self) -> int:
        """Number of nodes in the last graph, or 0 if none."""
        return len(self.prev_coords) if self.prev_coords is not None else 0
