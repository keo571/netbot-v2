"""
Data models for RAG Orchestrator.

Defines the core data structures for hybrid RAG queries, responses,
and reliability metrics.
"""

from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field
import uuid

from ...shared.models import TimestampMixin


class QueryType(str, Enum):
    """Types of RAG queries."""
    SEMANTIC_SEARCH = "semantic_search"
    GRAPH_TRAVERSAL = "graph_traversal"  
    HYBRID_FUSION = "hybrid_fusion"
    CONVERSATIONAL = "conversational"
    ANALYTICAL = "analytical"


class ProcessingMode(str, Enum):
    """RAG processing modes."""
    FAST = "fast"  # Vector search only
    BALANCED = "balanced"  # Hybrid with limited graph expansion
    COMPREHENSIVE = "comprehensive"  # Full hybrid with deep graph traversal
    INTERACTIVE = "interactive"  # Optimized for conversational AI


class ReliabilityLevel(str, Enum):
    """Reliability assessment levels."""
    HIGH = "high"  # >0.8 confidence
    MEDIUM = "medium"  # 0.5-0.8 confidence
    LOW = "low"  # 0.2-0.5 confidence
    UNCERTAIN = "uncertain"  # <0.2 confidence


class RAGQuery(BaseModel):
    """RAG query request model."""
    
    query_id: str = Field(default_factory=lambda: f"query_{uuid.uuid4().hex[:12]}")
    query_text: str = Field(..., description="User's natural language query")
    query_type: QueryType = Field(default=QueryType.HYBRID_FUSION)
    processing_mode: ProcessingMode = Field(default=ProcessingMode.BALANCED)
    
    # Context information
    session_id: Optional[str] = Field(None, description="Session ID for conversational context")
    user_id: Optional[str] = Field(None, description="User ID for personalization")
    diagram_id: Optional[str] = Field(None, description="Specific diagram to search")
    
    # Search parameters
    top_k: int = Field(default=5, ge=1, le=50)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    
    # Simple filtering
    categories: Optional[List[str]] = Field(None, description="Filter by content categories (e.g., 'security', 'networking', 'beginner')")
    
    # Advanced options (simplified)
    include_visualizations: bool = Field(default=True)
    
    # System parameters
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)


class SourceReference(BaseModel):
    """Reference to source material."""
    
    source_id: str = Field(..., description="Unique source identifier")
    source_type: str = Field(..., description="Type of source (document, chunk, graph_node)")
    title: str = Field(..., description="Source title or name")
    content_excerpt: str = Field(..., description="Relevant content excerpt")
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    
    # Location information
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    node_id: Optional[str] = None
    diagram_id: Optional[str] = None


class ConfidenceMetrics(BaseModel):
    """Simplified confidence and reliability metrics."""
    
    # Core metrics (what users actually care about)
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    reliability_level: ReliabilityLevel = Field(..., description="Categorized reliability level")
    
    # Optional detailed breakdown (for advanced users/debugging)
    source_coverage: Optional[float] = Field(None, ge=0.0, le=1.0, description="How well sources cover the query")
    response_grounding: Optional[float] = Field(None, ge=0.0, le=1.0, description="How well response is grounded in evidence")
    
    # Simple indicators
    has_gaps: bool = Field(default=False, description="Whether information gaps were identified")
    sources_used: int = Field(..., description="Number of sources used in response")


class RAGContext(BaseModel):
    """Simplified context for response generation."""
    
    # Retrieved content (essential)
    text_results: List[Dict[str, Any]] = Field(default_factory=list)
    graph_data: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Conversation context (if available)
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Simple quality indicator
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall context quality")


class VisualizationData(BaseModel):
    """Simplified visualization data."""
    
    visualization_id: str = Field(default_factory=lambda: f"viz_{uuid.uuid4().hex[:8]}")
    visualization_type: str = Field(..., description="Type of visualization")
    title: str = Field(..., description="Visualization title")
    
    # Core graph data
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Rendered visualization (system chooses best format)
    rendered_content: Optional[str] = Field(None, description="Rendered visualization as HTML, SVG, or JSON string")
    content_type: str = Field(default="html", description="Format of rendered content: 'html', 'svg', or 'json'")


class RAGResponse(BaseModel, TimestampMixin):
    """Simplified RAG response."""
    
    # Essential response information
    query_id: str = Field(..., description="Original query ID")
    response_id: str = Field(default_factory=lambda: f"resp_{uuid.uuid4().hex[:12]}")
    
    # Core content
    response_text: str = Field(..., description="Generated response text")
    sources: List[SourceReference] = Field(default_factory=list)
    
    # Quality assessment
    confidence_metrics: ConfidenceMetrics = Field(..., description="Confidence assessment")
    
    # Optional enhancements
    visualizations: List[VisualizationData] = Field(default_factory=list)
    suggested_follow_ups: List[str] = Field(default_factory=list)
    
    # System info (simplified)
    processing_mode: ProcessingMode = Field(..., description="Processing mode used")
    processing_time_ms: float = Field(..., description="Total processing time")


class ProcessingStatus(BaseModel, TimestampMixin):
    """Simplified processing status for async operations."""
    
    operation_id: str = Field(..., description="Unique operation identifier")
    status: str = Field(..., description="Current status (started, processing, completed, failed)")
    progress: float = Field(default=0.0, ge=0.0, le=100.0, description="Completion percentage")
    
    # Results (when completed)
    result: Optional[Union[RAGResponse, Dict[str, Any]]] = None
    error: Optional[str] = None


class BatchRAGRequest(BaseModel):
    """Simplified batch processing for testing and system operations."""
    
    batch_id: str = Field(default_factory=lambda: f"batch_{uuid.uuid4().hex[:12]}")
    queries: List[RAGQuery] = Field(..., min_items=1, max_items=100)
    
    # Simple processing options
    max_concurrent: int = Field(default=5, ge=1, le=10)
    fail_fast: bool = Field(default=False, description="Stop on first error")


class BatchRAGResponse(BaseModel, TimestampMixin):
    """Simplified batch processing response."""
    
    batch_id: str = Field(..., description="Batch identifier")
    total_queries: int = Field(..., description="Total number of queries")
    
    # Results
    successful_responses: List[RAGResponse] = Field(default_factory=list)
    failed_queries: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Essential statistics
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Percentage of successful queries")
    avg_processing_time_ms: float = Field(..., description="Average processing time per query")
    total_processing_time_ms: float = Field(..., description="Total batch processing time")
    
    # Optional quality summary
    avg_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Average confidence of successful responses")