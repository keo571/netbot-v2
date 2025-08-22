"""
Advanced/experimental embedding features for NetBot-v2.

These features provide hybrid RAG capabilities including:
- Document chunking and processing
- Vector database integration (ChromaDB)
- Advanced retrieval strategies

Note: These features require optional dependencies and are experimental.
Dependencies: chromadb, sentence-transformers
"""

# Try to import advanced components (requires optional dependencies)
try:
    from .hybrid_manager import HybridEmbeddingManager
    from .chunking import HybridChunker, DiagramChunk, ChunkType
    from .vector_stores import BaseVectorStore, ChromaVectorStore
    
    __all__ = [
        'HybridEmbeddingManager',   # Main hybrid RAG interface
        'HybridChunker',            # Document chunking
        'DiagramChunk', 
        'ChunkType',
        'BaseVectorStore',          # Vector database abstraction
        'ChromaVectorStore'         # ChromaDB implementation
    ]
    
    # Mark as available
    ADVANCED_FEATURES_AVAILABLE = True
    
except ImportError as e:
    # Advanced dependencies not available
    __all__ = []
    ADVANCED_FEATURES_AVAILABLE = False
    _IMPORT_ERROR = str(e)

def check_advanced_dependencies():
    """Check if advanced embedding features are available."""
    if ADVANCED_FEATURES_AVAILABLE:
        return True, "Advanced embedding features available"
    else:
        return False, f"Advanced features not available: {_IMPORT_ERROR}"