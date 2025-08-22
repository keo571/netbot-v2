"""
Canonical graph data models for the entire netbot-v2 system.

These models represent the core entities stored in Neo4j and used across:
- diagram_processing: Creates these entities from images
- graph_rag: Queries and manipulates these entities for search
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph"""
    id: str
    label: str
    type: str
    diagram_id: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass
class GraphRelationship:
    """Represents a relationship in the knowledge graph"""
    id: str
    source_id: str
    target_id: str
    type: str
    diagram_id: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass
class Shape:
    """Represents a detected shape in the image (used during diagram processing)"""
    type: str  # 'rectangle', 'circle', 'line', 'arrow'
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    properties: Dict[str, Any] = field(default_factory=dict)