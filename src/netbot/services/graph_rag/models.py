"""
Service models for GraphRAG operations.
"""

from typing import Dict, Any, List, Optional, Union
from enum import Enum
from pydantic import Field, validator

from ...shared.models.base import BaseModel, TimestampMixin
from ...shared.models.graph import GraphNode, GraphRelationship
from ...shared.models.retrieval import Query, RetrievalMetadata


class SearchMethod(str, Enum):
    """Available search methods."""
    VECTOR = "vector"
    CYPHER = "cypher"
    AUTO = "auto"
    HYBRID = "hybrid"


class VisualizationBackend(str, Enum):
    """Available visualization backends."""
    GRAPHVIZ = "graphviz"
    NETWORKX = "networkx"


class SearchRequest(BaseModel):
    """Request for graph search operations."""
    
    query: str = Field(..., description="Natural language search query")
    diagram_id: str = Field(..., description="Target diagram ID")
    
    # Search parameters
    method: SearchMethod = Field(default=SearchMethod.AUTO, description="Search method to use")
    top_k: int = Field(default=8, ge=1, le=100, description="Number of results to return")
    min_similarity: float = Field(default=0.1, ge=0.0, le=1.0, description="Minimum similarity threshold")
    
    # Context parameters
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    
    # Response options
    include_explanation: bool = Field(default=False, description="Generate explanation")
    detailed_explanation: bool = Field(default=False, description="Use detailed explanation format")
    include_visualization: bool = Field(default=False, description="Generate visualization")
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query text."""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()
    
    @validator('diagram_id')
    def validate_diagram_id(cls, v):
        """Validate diagram ID."""
        if not v or not v.strip():
            raise ValueError("Diagram ID cannot be empty")
        return v.strip()


class VisualizationRequest(BaseModel):
    """Request for graph visualization."""
    
    backend: VisualizationBackend = Field(default=VisualizationBackend.GRAPHVIZ, description="Visualization backend")
    layout: Optional[str] = Field(default=None, description="Layout algorithm")
    format: str = Field(default="svg", description="Output format (svg, png)")
    
    # Display options
    show_node_properties: bool = Field(default=True, description="Show node properties")
    show_edge_properties: bool = Field(default=True, description="Show relationship properties")
    node_color_by: Optional[str] = Field(default="type", description="Node coloring attribute")
    edge_color_by: Optional[str] = Field(default="type", description="Edge coloring attribute")
    
    # Size options
    width: int = Field(default=1200, ge=200, le=4000, description="Image width")
    height: int = Field(default=800, ge=200, le=4000, description="Image height")
    node_size: int = Field(default=1000, ge=100, le=5000, description="Node size")
    font_size: int = Field(default=12, ge=8, le=24, description="Font size")


class SearchResult(BaseModel, TimestampMixin):
    """Result of graph search operation."""
    
    # Request reference
    request: SearchRequest = Field(..., description="Original search request")
    
    # Search results
    nodes: List[GraphNode] = Field(default_factory=list, description="Retrieved nodes")
    relationships: List[GraphRelationship] = Field(default_factory=list, description="Retrieved relationships")
    
    # Search metadata
    metadata: RetrievalMetadata = Field(default_factory=RetrievalMetadata, description="Search metadata")
    
    # Generated content
    explanation: Optional[str] = Field(default=None, description="AI-generated explanation")
    visualization_data: Optional[Dict[str, Any]] = Field(default=None, description="Visualization JSON data for frontend rendering")
    
    # Quality metrics
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall relevance")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Result confidence")
    
    # Error handling
    success: bool = Field(default=True, description="Whether search succeeded")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    
    @property
    def result_count(self) -> int:
        """Total number of results."""
        return len(self.nodes) + len(self.relationships)
    
    @property
    def is_empty(self) -> bool:
        """Check if results are empty."""
        return len(self.nodes) == 0 and len(self.relationships) == 0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get result summary."""
        return {
            'query': self.request.query,
            'diagram_id': self.request.diagram_id,
            'method': self.request.method,
            'success': self.success,
            'node_count': len(self.nodes),
            'relationship_count': len(self.relationships),
            'relevance_score': self.relevance_score,
            'search_time_ms': self.metadata.search_time_ms,
            'created_at': self.created_at.isoformat(),
            'error_message': self.error_message
        }


class ExplanationRequest(BaseModel):
    """Request for generating explanations."""
    
    nodes: List[GraphNode] = Field(..., description="Nodes to explain")
    relationships: List[GraphRelationship] = Field(..., description="Relationships to explain")
    original_query: str = Field(..., description="Original search query")
    
    detailed: bool = Field(default=False, description="Generate detailed explanation")
    diagram_type: Optional[str] = Field(default=None, description="Diagram type hint")
    
    # AI parameters
    model_name: str = Field(default="gemini-2.0-flash-exp", description="AI model to use")
    temperature: float = Field(default=0.1, description="Generation temperature")
    max_tokens: int = Field(default=2048, description="Maximum response tokens")


class VisualizationResult(BaseModel):
    """Result of visualization generation - now returns JSON data instead of images."""
    
    visualization_data: Dict[str, Any] = Field(..., description="JSON visualization data for frontend rendering")
    visualization_type: str = Field(default="network_graph", description="Type of visualization")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Visualization metadata")
    
    success: bool = Field(default=True, description="Whether visualization succeeded")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")