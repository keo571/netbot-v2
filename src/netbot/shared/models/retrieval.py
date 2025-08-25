"""
Retrieval and search models for NetBot V2.

These models represent queries, results, and contexts used across
different retrieval systems (vector, graph, hybrid).
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Union
from pydantic import Field, validator

from .base import BaseModel, TimestampMixin
from .graph import GraphNode, GraphRelationship


class QueryType(str, Enum):
    """Types of queries supported by the system."""
    SEMANTIC = "semantic"          # Vector similarity search
    STRUCTURAL = "structural"      # Graph traversal
    HYBRID = "hybrid"             # Combined semantic + structural
    CONVERSATIONAL = "conversational"  # Context-aware chat


class SearchMethod(str, Enum):
    """Search methods available."""
    VECTOR = "vector"
    CYPHER = "cypher" 
    AUTO = "auto"
    HYBRID = "hybrid"


class Query(BaseModel, TimestampMixin):
    """
    Represents a search query with context and parameters.
    """
    
    text: str = Field(..., description="Query text")
    query_type: QueryType = Field(default=QueryType.HYBRID, description="Type of query")
    method: SearchMethod = Field(default=SearchMethod.AUTO, description="Search method to use")
    
    # Target constraints
    diagram_id: Optional[str] = Field(default=None, description="Target diagram ID")
    document_ids: List[str] = Field(default_factory=list, description="Target document IDs")
    
    # Search parameters
    top_k: int = Field(default=8, ge=1, le=100, description="Number of results to return")
    min_similarity: float = Field(default=0.1, ge=0.0, le=1.0, description="Minimum similarity threshold")
    
    # Context and session
    session_id: Optional[str] = Field(default=None, description="Conversation session ID")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    context_window: int = Field(default=5, ge=0, le=20, description="Context window size")
    
    # Query enhancement
    original_text: Optional[str] = Field(default=None, description="Original query before enhancement")
    intent: Optional[str] = Field(default=None, description="Detected query intent")
    entities: List[str] = Field(default_factory=list, description="Extracted entities")
    
    @validator('text')
    def validate_text(cls, v):
        """Validate query text."""
        if not v or not v.strip():
            raise ValueError("Query text cannot be empty")
        return v.strip()


class RetrievalMetadata(BaseModel):
    """
    Metadata about the retrieval process and results.
    """
    
    # Performance metrics
    search_time_ms: float = Field(default=0.0, description="Total search time")
    vector_search_time_ms: float = Field(default=0.0, description="Vector search time")
    graph_search_time_ms: float = Field(default=0.0, description="Graph search time")
    
    # Result quality
    total_candidates: int = Field(default=0, description="Total candidates found")
    filtered_results: int = Field(default=0, description="Results after filtering")
    average_similarity: float = Field(default=0.0, description="Average similarity score")
    
    # Source information
    sources_used: List[str] = Field(default_factory=list, description="Data sources used")
    embeddings_cache_hit: bool = Field(default=False, description="Whether embeddings were cached")
    
    # Quality flags
    is_complete: bool = Field(default=True, description="Whether search completed successfully")
    has_warnings: bool = Field(default=False, description="Whether there were warnings")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")


class SearchResult(BaseModel, TimestampMixin):
    """
    Represents search results with rich metadata.
    """
    
    # Core results
    nodes: List[GraphNode] = Field(default_factory=list, description="Retrieved nodes")
    relationships: List[GraphRelationship] = Field(default_factory=list, description="Retrieved relationships")
    
    # Query reference
    query: Query = Field(..., description="Original query")
    
    # Result metadata
    metadata: RetrievalMetadata = Field(default_factory=RetrievalMetadata, description="Retrieval metadata")
    
    # Generated content
    explanation: Optional[str] = Field(default=None, description="AI-generated explanation")
    visualization_data: Optional[str] = Field(default=None, description="Visualization (SVG/PNG base64)")
    
    # Quality scores
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall relevance score")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Result confidence")
    
    @property
    def result_count(self) -> int:
        """Total number of results (nodes + relationships)."""
        return len(self.nodes) + len(self.relationships)
    
    @property
    def is_empty(self) -> bool:
        """Check if results are empty."""
        return len(self.nodes) == 0 and len(self.relationships) == 0
    
    def filter_by_relevance(self, min_score: float) -> "SearchResult":
        """Filter results by minimum relevance score."""
        # For now, use node confidence as proxy for relevance
        filtered_nodes = [n for n in self.nodes if n.confidence_score >= min_score]
        
        # Keep relationships between filtered nodes
        retained_ids = {n.id for n in filtered_nodes}
        filtered_relationships = [
            r for r in self.relationships 
            if r.source_id in retained_ids and r.target_id in retained_ids
        ]
        
        return SearchResult(
            nodes=filtered_nodes,
            relationships=filtered_relationships,
            query=self.query,
            metadata=self.metadata,
            explanation=self.explanation,
            visualization_data=self.visualization_data,
            relevance_score=min_score,
            confidence_score=self.confidence_score,
        )


class Context(BaseModel):
    """
    Represents assembled context for response generation.
    
    This is used by LLMs to generate responses with proper grounding.
    """
    
    # Core context
    query: Query = Field(..., description="Original query")
    search_results: List[SearchResult] = Field(default_factory=list, description="Search results")
    
    # Conversation context
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list, description="Previous exchanges")
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    
    # Document context (for text RAG)
    text_chunks: List[Dict[str, Any]] = Field(default_factory=list, description="Relevant text chunks")
    document_metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    
    # Cross-modal links
    text_diagram_links: Dict[str, str] = Field(default_factory=dict, description="Text-diagram associations")
    
    # Context quality
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Context completeness")
    consistency_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Cross-modal consistency")
    
    @property
    def total_nodes(self) -> int:
        """Total nodes across all search results."""
        return sum(len(result.nodes) for result in self.search_results)
    
    @property
    def total_relationships(self) -> int:
        """Total relationships across all search results."""
        return sum(len(result.relationships) for result in self.search_results)
    
    @property
    def has_graph_data(self) -> bool:
        """Check if context has graph data."""
        return self.total_nodes > 0 or self.total_relationships > 0
    
    @property
    def has_text_data(self) -> bool:
        """Check if context has text data."""
        return len(self.text_chunks) > 0
    
    @property
    def is_multimodal(self) -> bool:
        """Check if context spans multiple modalities."""
        return self.has_graph_data and self.has_text_data
    
    def add_search_result(self, result: SearchResult) -> None:
        """Add a search result to the context."""
        self.search_results.append(result)
    
    def get_all_nodes(self) -> List[GraphNode]:
        """Get all nodes from all search results."""
        nodes = []
        for result in self.search_results:
            nodes.extend(result.nodes)
        return nodes
    
    def get_all_relationships(self) -> List[GraphRelationship]:
        """Get all relationships from all search results."""
        relationships = []
        for result in self.search_results:
            relationships.extend(result.relationships)
        return relationships