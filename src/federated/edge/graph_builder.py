"""
Graph Builder
-------------
Builds stable spatial interaction graphs from tracked persons for STGNN.

Key invariants for the real-time pipeline:
- Node ordering is deterministic across frames (sorted by track_id).
- Spatial edges are based on interaction radius, not forced kNN density.
- Local density reflects nearby crowding inside the interaction radius.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.spatial.distance import cdist


logger = logging.getLogger(__name__)

# Constants
MAX_NODES: int = 100
MAX_VELOCITY = 0.1
DENSITY_SATURATION_NEIGHBORS = 6

class GraphBuilder:
    """
    Build a padded radius-based graph from tracked persons.

    Stable node ordering is critical because the temporal buffer stacks
    node features by row index across frames.
    """
    
    def __init__(self, radius: float = 0.05, min_nodes: int = 2, max_nodes: int = MAX_NODES) -> None:
        if radius <= 0:
            raise ValueError(f"radius must be > 0, got {radius}")
        self.radius = float(radius)
        self.min_nodes = min_nodes
        self.max_nodes = int(max_nodes)

    def order_tracked_persons(self, tracked_persons: List[Any]) -> List[Any]:
        """
        Return a deterministic node order for temporal consistency.

        ByteTrack IDs are stable across frames, so sorting by track_id keeps
        each person's feature row much more consistent over time.
        """
        return sorted(
            tracked_persons,
            key=lambda person: (
                getattr(person, "track_id", float("inf")),
                getattr(person, "cx", 0.0),
                getattr(person, "cy", 0.0),
            ),
        )
    
    def build_graph(
        self,
        tracked_persons: List[Any],
        frame_shape: Tuple[int, int],
        prev_positions: Dict[int, Tuple[float, float]] = None,
    ) -> Optional[Dict[str, np.ndarray]]:
        if not tracked_persons or len(tracked_persons) < self.min_nodes:
            return None
            
        if prev_positions is None:
            prev_positions = {}
        
        h, w = frame_shape
        if h <= 0 or w <= 0:
            return None
        
        n_actual = min(len(tracked_persons), self.max_nodes)
        persons = self.order_tracked_persons(tracked_persons)[:n_actual]
        
        coords = np.zeros((n_actual, 2), dtype=np.float32)
        bbox_areas = np.zeros(n_actual, dtype=np.float32)
        track_ids = np.zeros(n_actual, dtype=np.int64)
        
        for i, p in enumerate(persons):
            coords[i, 0] = p.cx / float(w)
            coords[i, 1] = p.cy / float(h)
            bw = (p.x2 - p.x1) / float(w)
            bh = (p.y2 - p.y1) / float(h)
            bbox_areas[i] = bw * bh
            track_ids[i] = p.track_id
            
        velocity = np.zeros((n_actual, 2), dtype=np.float32)
        for i, p in enumerate(persons):
            tid = p.track_id
            if tid in prev_positions:
                prev_x, prev_y = prev_positions[tid]
                dx = coords[i, 0] - prev_x
                dy = coords[i, 1] - prev_y
                dx = float(np.clip(dx, -MAX_VELOCITY, MAX_VELOCITY))
                dy = float(np.clip(dy, -MAX_VELOCITY, MAX_VELOCITY))
                velocity[i, 0] = dx
                velocity[i, 1] = dy
                
        distance_matrix = cdist(coords, coords)
        speed = np.sqrt(velocity[:, 0] ** 2 + velocity[:, 1] ** 2)
        heading = np.arctan2(velocity[:, 1], velocity[:, 0]) / np.pi
        
        edge_index = self._build_radius_edges(distance_matrix)
        local_density = self._compute_density(distance_matrix)
        
        features = np.hstack([
            coords,                 # [N, 2]
            velocity,               # [N, 2]
            speed[:, None],         # [N, 1]
            heading[:, None],       # [N, 1]
            local_density[:, None], # [N, 1]
            bbox_areas[:, None],    # [N, 1]
        ]).astype(np.float32)
        
        n_feat = features.shape[1]
        x_padded = np.zeros((self.max_nodes, n_feat), dtype=np.float32)
        mask = np.zeros(self.max_nodes, dtype=np.float32)
        
        x_padded[:n_actual] = features
        mask[:n_actual] = 1.0
        
        return {
            "x": x_padded,
            "mask": mask,
            "edge_index": edge_index,
            "track_ids": track_ids,
        }
        
    def _build_radius_edges(self, distance_matrix: np.ndarray) -> np.ndarray:
        n = distance_matrix.shape[0]
        if n < 2:
            return np.zeros((2, 0), dtype=np.int64)

        adjacency = (
            (distance_matrix > 0.0)
            & np.isfinite(distance_matrix)
            & (distance_matrix <= self.radius)
        )

        rows, cols = np.where(adjacency)
        if rows.size == 0:
            return np.zeros((2, 0), dtype=np.int64)

        return np.stack([rows, cols], axis=0).astype(np.int64)
        
    def _compute_density(self, distance_matrix: np.ndarray) -> np.ndarray:
        n = distance_matrix.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.float32)

        neighbor_mask = (
            (distance_matrix > 0.0)
            & np.isfinite(distance_matrix)
            & (distance_matrix <= self.radius)
        )
        neighbor_count = neighbor_mask.sum(axis=1).astype(np.float32)

        density = neighbor_count / float(DENSITY_SATURATION_NEIGHBORS)
        return np.clip(density, 0.0, 1.0)
        
    def reset(self) -> None:
        pass
    
    @property
    def current_node_count(self) -> int:
        return 0
