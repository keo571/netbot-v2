"""
Data models for chunking in hybrid RAG system.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


class ChunkType(Enum):
    """Types of chunks in the hybrid RAG system."""
    DIAGRAM_WITH_CONTEXT = "diagram_with_context"  # Diagram ID + surrounding text
    PURE_TEXT = "pure_text"                        # Text without diagram reference
    DIAGRAM_METADATA = "diagram_metadata"          # Diagram description/summary


@dataclass
class DiagramChunk:
    """
    A semantic chunk that combines diagram information with contextual text.
    
    This is the core unit for hybrid RAG retrieval - when a chunk is found
    relevant, it can trigger both text response and graph-based retrieval
    using the diagram_id.
    """
    
    # Core identifiers
    chunk_id: str                           # Unique chunk identifier
    diagram_id: Optional[str] = None        # Diagram ID if chunk references a diagram
    
    # Content
    text_content: str = ""                  # The actual text content 
    diagram_context: str = ""               # Description/context about the diagram
    
    # Metadata
    chunk_type: ChunkType = ChunkType.PURE_TEXT
    source_document: Optional[str] = None   # Original document/file name
    page_number: Optional[int] = None       # Page number in source document
    
    # Position information
    text_before_diagram: str = ""           # Text appearing before diagram
    text_after_diagram: str = ""            # Text appearing after diagram
    
    # Embedding and search
    embedding: Optional[List[float]] = None # Vector embedding of combined content
    embedding_model: str = "sentence-transformers" # Model used for embedding
    
    # Additional metadata
    properties: Dict[str, Any] = None       # Additional flexible properties
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
    
    def get_full_content(self) -> str:
        """
        Get the complete textual content for embedding.
        
        Combines all text fields into a coherent string that represents
        the semantic meaning of this chunk.
        """
        parts = []
        
        # Add text before diagram
        if self.text_before_diagram.strip():
            parts.append(self.text_before_diagram.strip())
        
        # Add diagram context/description
        if self.diagram_context.strip():
            parts.append(f"Diagram: {self.diagram_context.strip()}")
        
        # Add main text content
        if self.text_content.strip():
            parts.append(self.text_content.strip())
        
        # Add text after diagram
        if self.text_after_diagram.strip():
            parts.append(self.text_after_diagram.strip())
        
        return " ".join(parts)
    
    def has_diagram_reference(self) -> bool:
        """Check if this chunk references a diagram."""
        return self.diagram_id is not None and self.diagram_id.strip() != ""
    
    def get_metadata_dict(self) -> Dict[str, Any]:
        """Get chunk metadata as dictionary for storage."""
        return {
            "chunk_id": self.chunk_id,
            "diagram_id": self.diagram_id,
            "chunk_type": self.chunk_type.value,
            "source_document": self.source_document,
            "page_number": self.page_number,
            "embedding_model": self.embedding_model,
            "has_diagram": self.has_diagram_reference(),
            **self.properties
        }