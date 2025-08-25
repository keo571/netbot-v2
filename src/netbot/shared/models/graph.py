"""
Graph data models for NetBot V2.

These models represent the core entities stored in Neo4j and used across:
- diagram_processing: Creates these entities from images
- graph_rag: Queries and manipulates these entities for search
- text_rag: Links these entities with textual content
- evaluation: Tests and validates these structures
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pydantic import Field, field_validator

from .base import BaseModel, TimestampMixin, MetadataMixin


class GraphNode(BaseModel, TimestampMixin, MetadataMixin):
    """
    Represents a node in the knowledge graph.
    
    A node represents an entity (e.g., server, process, decision) extracted
    from a diagram or document.
    """
    
    id: str = Field(..., description="Unique identifier for the node")
    label: str = Field(..., description="Human-readable label")
    type: str = Field(..., description="Node type (e.g., 'Server', 'Process')")
    diagram_id: str = Field(..., description="ID of the source diagram")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional node properties")
    embedding: Optional[np.ndarray] = Field(default=None, description="Vector embedding for similarity search")
    
    # Validation and confidence scores
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence")
    source_type: str = Field(default="diagram", description="Source type (diagram, document, etc.)")
    
    @field_validator('embedding', mode='before')
    @classmethod
    def validate_embedding(cls, v):
        """Validate embedding is numpy array with correct shape."""
        if v is not None and not isinstance(v, np.ndarray):
            raise ValueError("Embedding must be a numpy array")
        return v
    
    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        """Normalize node type."""
        return v.strip().title() if v else v
    
    class Config:
        # Allow numpy arrays in Pydantic models
        arbitrary_types_allowed = True
        # JSON serialization for numpy arrays
        json_encoders = {
            np.ndarray: lambda x: x.tolist() if x is not None else None
        }
    
    def add_property(self, key: str, value: Any) -> None:
        """Add a property to the node."""
        self.properties[key] = value
    
    def get_property(self, key: str, default: Any = None) -> Any:
        """Get a property value."""
        return self.properties.get(key, default)
    
    def to_neo4j_dict(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for Neo4j storage."""
        data = {
            'id': self.id,
            'label': self.label,
            'type': self.type,
            'diagram_id': self.diagram_id,
            'confidence_score': self.confidence_score,
            'source_type': self.source_type,
            'created_at': self.created_at.isoformat(),
        }
        
        # Add properties
        data.update(self.properties)
        
        # Add embedding as list if present
        if self.embedding is not None:
            data['embedding'] = self.embedding.tolist()
        
        # Add metadata
        if self.metadata:
            data.update({f"meta_{k}": v for k, v in self.metadata.items()})
        
        return data


class GraphRelationship(BaseModel, TimestampMixin, MetadataMixin):
    """
    Represents a relationship in the knowledge graph.
    
    A relationship connects two nodes and represents an interaction,
    flow, or association between entities.
    """
    
    id: str = Field(..., description="Unique identifier for the relationship")
    source_id: str = Field(..., description="ID of the source node")
    target_id: str = Field(..., description="ID of the target node")
    type: str = Field(..., description="Relationship type (e.g., 'CONNECTS_TO', 'FLOWS_TO')")
    diagram_id: str = Field(..., description="ID of the source diagram")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional relationship properties")
    
    # Validation and confidence scores
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence")
    source_type: str = Field(default="diagram", description="Source type (diagram, document, etc.)")
    
    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        """Normalize relationship type."""
        return v.strip().upper().replace(' ', '_') if v else v
    
    def add_property(self, key: str, value: Any) -> None:
        """Add a property to the relationship."""
        self.properties[key] = value
    
    def get_property(self, key: str, default: Any = None) -> Any:
        """Get a property value."""
        return self.properties.get(key, default)
    
    def to_neo4j_dict(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for Neo4j storage."""
        data = {
            'id': self.id,
            'type': self.type,
            'diagram_id': self.diagram_id,
            'confidence_score': self.confidence_score,
            'source_type': self.source_type,
            'created_at': self.created_at.isoformat(),
        }
        
        # Add properties
        data.update(self.properties)
        
        # Add metadata
        if self.metadata:
            data.update({f"meta_{k}": v for k, v in self.metadata.items()})
        
        return data


class GraphResult(BaseModel):
    """
    Represents a complete graph search result.
    
    Contains nodes, relationships, and metadata about the search operation.
    """
    
    nodes: List[GraphNode] = Field(default_factory=list, description="Retrieved nodes")
    relationships: List[GraphRelationship] = Field(default_factory=list, description="Retrieved relationships")
    
    # Search metadata
    query: str = Field(..., description="Original search query")
    diagram_id: str = Field(..., description="Target diagram ID")
    method: str = Field(default="auto", description="Search method used")
    
    # Performance metrics
    total_nodes: int = Field(default=0, description="Total nodes found")
    total_relationships: int = Field(default=0, description="Total relationships found")
    search_time_ms: float = Field(default=0.0, description="Search time in milliseconds")
    
    # Quality metrics
    average_confidence: float = Field(default=0.0, description="Average confidence score")
    min_confidence: float = Field(default=0.0, description="Minimum confidence score")
    max_confidence: float = Field(default=0.0, description="Maximum confidence score")
    
    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        self.total_nodes = len(self.nodes)
        self.total_relationships = len(self.relationships)
        
        if self.nodes:
            confidences = [node.confidence_score for node in self.nodes]
            self.average_confidence = sum(confidences) / len(confidences)
            self.min_confidence = min(confidences)
            self.max_confidence = max(confidences)
    
    def filter_by_confidence(self, min_confidence: float) -> "GraphResult":
        """Filter results by minimum confidence score."""
        filtered_nodes = [n for n in self.nodes if n.confidence_score >= min_confidence]
        
        # Filter relationships to only include those between retained nodes
        retained_node_ids = {n.id for n in filtered_nodes}
        filtered_relationships = [
            r for r in self.relationships 
            if r.source_id in retained_node_ids and r.target_id in retained_node_ids
        ]
        
        return GraphResult(
            nodes=filtered_nodes,
            relationships=filtered_relationships,
            query=self.query,
            diagram_id=self.diagram_id,
            method=self.method,
            search_time_ms=self.search_time_ms,
        )
    
    def get_node_by_id(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by its ID."""
        return next((n for n in self.nodes if n.id == node_id), None)
    
    def get_relationships_for_node(self, node_id: str) -> List[GraphRelationship]:
        """Get all relationships involving a specific node."""
        return [
            r for r in self.relationships 
            if r.source_id == node_id or r.target_id == node_id
        ]


class Shape(BaseModel):
    """
    Represents a detected shape in an image during diagram processing.
    
    Used as an intermediate representation before creating GraphNode objects.
    """
    
    type: str = Field(..., description="Shape type (rectangle, circle, line, arrow)")
    bbox: Tuple[int, int, int, int] = Field(..., description="Bounding box (x1, y1, x2, y2)")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Shape properties")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Detection confidence")
    
    @field_validator('bbox')
    @classmethod
    def validate_bbox(cls, v):
        """Validate bounding box has correct format."""
        if len(v) != 4:
            raise ValueError("Bounding box must have 4 coordinates (x1, y1, x2, y2)")
        x1, y1, x2, y2 = v
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid bounding box coordinates")
        return v
    
    @property
    def area(self) -> int:
        """Calculate shape area."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    @property
    def center(self) -> Tuple[float, float]:
        """Calculate shape center point."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)