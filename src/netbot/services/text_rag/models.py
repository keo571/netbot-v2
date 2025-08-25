"""
TextRAG data models for NetBot V2.

Defines all data structures for document management, chunking,
vector storage, and semantic search operations.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

import numpy as np
from pydantic import Field, field_validator

from ...shared.models.base import BaseModel, TimestampMixin


class DocumentType(str, Enum):
    """Types of documents that can be processed."""
    TEXT = "text"
    PDF = "pdf"
    MARKDOWN = "markdown"
    HTML = "html"
    WORD = "word"
    POWERPOINT = "powerpoint"
    CSV = "csv"
    JSON = "json"


class DocumentStatus(str, Enum):
    """Processing status of documents."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


class ChunkingStrategy(str, Enum):
    """Available text chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SENTENCE_BOUNDARY = "sentence_boundary"
    PARAGRAPH_BOUNDARY = "paragraph_boundary"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    RECURSIVE_CHARACTER = "recursive_character"


class SearchMethod(str, Enum):
    """Available search methods for retrieval."""
    VECTOR_SIMILARITY = "vector_similarity"
    BM25_LEXICAL = "bm25_lexical"
    HYBRID_FUSION = "hybrid_fusion"
    CONTEXT_AWARE = "context_aware"


class Document(BaseModel, TimestampMixin):
    """
    Represents a document in the TextRAG system.
    
    Documents are the primary units of textual content that get processed,
    chunked, and made available for semantic search.
    """
    
    document_id: str = Field(default_factory=lambda: f"doc_{uuid.uuid4().hex[:12]}", description="Unique document identifier")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Raw document content")
    document_type: DocumentType = Field(..., description="Type of document")
    
    # Source information
    source_path: Optional[str] = Field(default=None, description="Original file path")
    source_url: Optional[str] = Field(default=None, description="Source URL if applicable")
    author: Optional[str] = Field(default=None, description="Document author")
    
    # Processing metadata
    status: DocumentStatus = Field(default=DocumentStatus.PENDING, description="Processing status")
    file_size: int = Field(default=0, description="File size in bytes")
    character_count: int = Field(default=0, description="Total character count")
    word_count: int = Field(default=0, description="Estimated word count")
    
    # Content organization
    tags: List[str] = Field(default_factory=list, description="Document tags for organization")
    categories: List[str] = Field(default_factory=list, description="Document categories")
    diagram_references: List[str] = Field(default_factory=list, description="Referenced diagram IDs")
    
    # Processing configuration
    chunking_strategy: ChunkingStrategy = Field(default=ChunkingStrategy.RECURSIVE_CHARACTER, description="Text chunking strategy")
    chunk_size: int = Field(default=1000, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    
    # Quality and validation
    processing_errors: List[str] = Field(default_factory=list, description="Processing error messages")
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Content quality score")
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional document metadata")
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        """Ensure content is not empty."""
        if not v or not v.strip():
            raise ValueError("Document content cannot be empty")
        return v.strip()
    
    @field_validator('chunk_size')
    @classmethod
    def validate_chunk_size(cls, v):
        """Ensure chunk size is reasonable."""
        if v < 100 or v > 10000:
            raise ValueError("Chunk size must be between 100 and 10000 characters")
        return v
    
    def __post_init__(self):
        """Calculate derived fields after initialization."""
        if self.content:
            self.character_count = len(self.content)
            self.word_count = len(self.content.split())
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the document."""
        if tag and tag not in self.tags:
            self.tags.append(tag)
    
    def add_category(self, category: str) -> None:
        """Add a category to the document."""
        if category and category not in self.categories:
            self.categories.append(category)
    
    def link_to_diagram(self, diagram_id: str) -> None:
        """Link this document to a diagram."""
        if diagram_id and diagram_id not in self.diagram_references:
            self.diagram_references.append(diagram_id)
    
    def mark_processing_error(self, error: str) -> None:
        """Record a processing error."""
        if error and error not in self.processing_errors:
            self.processing_errors.append(error)
            self.status = DocumentStatus.FAILED


class DocumentChunk(BaseModel, TimestampMixin):
    """
    Represents a chunk of text from a document.
    
    Chunks are the atomic units for vector storage and retrieval.
    They maintain references to their source document and context.
    """
    
    chunk_id: str = Field(default_factory=lambda: f"chunk_{uuid.uuid4().hex[:12]}", description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk text content")
    
    # Position and context
    chunk_index: int = Field(..., description="Position in document (0-based)")
    start_char: int = Field(default=0, description="Starting character position in document")
    end_char: int = Field(default=0, description="Ending character position in document")
    
    # Content metadata
    character_count: int = Field(default=0, description="Number of characters in chunk")
    word_count: int = Field(default=0, description="Number of words in chunk")
    sentence_count: int = Field(default=0, description="Number of sentences in chunk")
    
    # Vector representation
    embedding: Optional[np.ndarray] = Field(default=None, description="Vector embedding for similarity search")
    embedding_model: Optional[str] = Field(default=None, description="Model used for embedding generation")
    
    # Context and relationships
    previous_chunk_id: Optional[str] = Field(default=None, description="Previous chunk ID for context")
    next_chunk_id: Optional[str] = Field(default=None, description="Next chunk ID for context")
    section_title: Optional[str] = Field(default=None, description="Section or heading title")
    
    # Search optimization
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    named_entities: List[str] = Field(default_factory=list, description="Identified named entities")
    topics: List[str] = Field(default_factory=list, description="Identified topics")
    
    # Quality metrics
    coherence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Text coherence score")
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Chunk relevance score")
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional chunk metadata")
    
    class Config:
        # Allow numpy arrays in Pydantic models
        arbitrary_types_allowed = True
        # JSON serialization for numpy arrays
        json_encoders = {
            np.ndarray: lambda x: x.tolist() if x is not None else None
        }
    
    @field_validator('embedding', mode='before')
    @classmethod
    def validate_embedding(cls, v):
        """Validate embedding is numpy array with correct shape."""
        if v is not None and not isinstance(v, np.ndarray):
            raise ValueError("Embedding must be a numpy array")
        return v
    
    def __post_init__(self):
        """Calculate derived fields after initialization."""
        if self.content:
            self.character_count = len(self.content)
            self.word_count = len(self.content.split())
            # Simple sentence count (could be improved with NLTK)
            self.sentence_count = len([s for s in self.content.split('.') if s.strip()])
    
    def get_context_window(self, before_chars: int = 200, after_chars: int = 200) -> str:
        """Get contextual text around this chunk."""
        # This would be implemented to fetch context from the parent document
        # For now, return the chunk content itself
        return self.content
    
    def calculate_similarity(self, other_embedding: np.ndarray) -> float:
        """Calculate cosine similarity with another embedding."""
        if self.embedding is None or other_embedding is None:
            return 0.0
        
        # Cosine similarity calculation
        dot_product = np.dot(self.embedding, other_embedding)
        norm_a = np.linalg.norm(self.embedding)
        norm_b = np.linalg.norm(other_embedding)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def to_vector_dict(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for vector storage."""
        return {
            'id': self.chunk_id,
            'document_id': self.document_id,
            'content': self.content,
            'embedding': self.embedding.tolist() if self.embedding is not None else None,
            'metadata': {
                'chunk_index': self.chunk_index,
                'character_count': self.character_count,
                'word_count': self.word_count,
                'section_title': self.section_title,
                'keywords': self.keywords,
                'topics': self.topics,
                'created_at': self.created_at.isoformat(),
                **self.metadata
            }
        }


class SearchQuery(BaseModel):
    """
    Represents a search query for text retrieval.
    
    Encapsulates all parameters needed for semantic search across
    document chunks with optional context awareness.
    """
    
    query_id: str = Field(default_factory=lambda: f"query_{uuid.uuid4().hex[:8]}", description="Unique query identifier")
    text: str = Field(..., description="Search query text")
    
    # Search parameters
    method: SearchMethod = Field(default=SearchMethod.HYBRID_FUSION, description="Search method to use")
    top_k: int = Field(default=5, description="Number of results to return")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    
    # Filtering options
    document_ids: Optional[List[str]] = Field(default=None, description="Limit search to specific documents")
    categories: Optional[List[str]] = Field(default=None, description="Filter by document categories")
    tags: Optional[List[str]] = Field(default=None, description="Filter by document tags")
    date_from: Optional[datetime] = Field(default=None, description="Filter documents from date")
    date_to: Optional[datetime] = Field(default=None, description="Filter documents to date")
    
    # Context-aware search
    session_id: Optional[str] = Field(default=None, description="Context Manager session for conversational search")
    user_id: Optional[str] = Field(default=None, description="User ID for personalized search")
    conversation_context: Optional[List[str]] = Field(default=None, description="Recent conversation messages")
    
    # Reranking and fusion
    enable_reranking: bool = Field(default=True, description="Enable result reranking")
    fusion_weights: Dict[str, float] = Field(
        default_factory=lambda: {"vector": 0.7, "lexical": 0.3},
        description="Weights for hybrid search fusion"
    )
    
    # Additional parameters
    include_metadata: bool = Field(default=True, description="Include chunk metadata in results")
    expand_context: bool = Field(default=False, description="Include adjacent chunks for context")
    
    @field_validator('top_k')
    @classmethod
    def validate_top_k(cls, v):
        """Ensure top_k is reasonable."""
        if v < 1 or v > 100:
            raise ValueError("top_k must be between 1 and 100")
        return v
    
    @field_validator('fusion_weights')
    @classmethod
    def validate_fusion_weights(cls, v):
        """Ensure fusion weights sum to 1.0."""
        if abs(sum(v.values()) - 1.0) > 0.01:
            raise ValueError("Fusion weights must sum to 1.0")
        return v


class SearchResult(BaseModel):
    """
    Represents a search result from text retrieval.
    
    Contains the matched chunk, similarity scores, and context
    information for result presentation and further processing.
    """
    
    # Result identification
    chunk_id: str = Field(..., description="Matched chunk ID")
    document_id: str = Field(..., description="Source document ID")
    content: str = Field(..., description="Chunk content")
    
    # Scoring and ranking
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Vector similarity score")
    lexical_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Lexical match score")
    fusion_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Combined fusion score")
    final_rank: int = Field(default=0, description="Final ranking position")
    
    # Context and metadata
    document_title: str = Field(default="", description="Source document title")
    section_title: Optional[str] = Field(default=None, description="Section title if available")
    chunk_index: int = Field(default=0, description="Position in document")
    
    # Content preview
    context_before: Optional[str] = Field(default=None, description="Text before the chunk")
    context_after: Optional[str] = Field(default=None, description="Text after the chunk")
    highlighted_content: Optional[str] = Field(default=None, description="Content with query terms highlighted")
    
    # Additional information
    keywords: List[str] = Field(default_factory=list, description="Relevant keywords")
    topics: List[str] = Field(default_factory=list, description="Identified topics")
    named_entities: List[str] = Field(default_factory=list, description="Named entities")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional result metadata")
    
    def get_preview(self, max_length: int = 200) -> str:
        """Get a preview of the content with ellipsis if needed."""
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length].rstrip() + "..."
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'chunk_id': self.chunk_id,
            'document_id': self.document_id,
            'document_title': self.document_title,
            'content': self.content,
            'similarity_score': self.similarity_score,
            'lexical_score': self.lexical_score,
            'fusion_score': self.fusion_score,
            'rank': self.final_rank,
            'section_title': self.section_title,
            'keywords': self.keywords,
            'topics': self.topics,
            'preview': self.get_preview(),
            'metadata': self.metadata
        }


class SearchResultSet(BaseModel):
    """
    Represents a complete set of search results.
    
    Contains all results from a search query along with metadata
    about the search operation and performance metrics.
    """
    
    # Query information
    query_id: str = Field(..., description="Original query ID")
    query_text: str = Field(..., description="Original query text")
    search_method: SearchMethod = Field(..., description="Search method used")
    
    # Results
    results: List[SearchResult] = Field(default_factory=list, description="Search results")
    total_results: int = Field(default=0, description="Total number of results found")
    returned_results: int = Field(default=0, description="Number of results returned")
    
    # Performance metrics
    search_time_ms: float = Field(default=0.0, description="Search execution time")
    vector_search_time_ms: float = Field(default=0.0, description="Vector search time")
    lexical_search_time_ms: float = Field(default=0.0, description="Lexical search time")
    reranking_time_ms: float = Field(default=0.0, description="Reranking time")
    
    # Search metadata
    documents_searched: int = Field(default=0, description="Number of documents searched")
    chunks_searched: int = Field(default=0, description="Number of chunks searched")
    filters_applied: List[str] = Field(default_factory=list, description="Applied search filters")
    
    # Context information
    session_id: Optional[str] = Field(default=None, description="Associated session ID")
    user_id: Optional[str] = Field(default=None, description="User who performed search")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Search timestamp")
    
    def __post_init__(self):
        """Calculate derived metrics."""
        self.returned_results = len(self.results)
    
    def get_top_results(self, n: int = 5) -> List[SearchResult]:
        """Get the top N results."""
        return sorted(self.results, key=lambda r: r.fusion_score, reverse=True)[:n]
    
    def filter_by_score(self, min_score: float) -> List[SearchResult]:
        """Filter results by minimum fusion score."""
        return [r for r in self.results if r.fusion_score >= min_score]
    
    def get_unique_documents(self) -> List[str]:
        """Get list of unique document IDs in results."""
        return list(set(r.document_id for r in self.results))