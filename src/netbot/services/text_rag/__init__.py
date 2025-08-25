"""
TextRAG Service for NetBot V2.

Provides document ingestion, semantic search, and text-based retrieval
capabilities that integrate with the Context Manager for conversational AI.

This service completes the hybrid RAG architecture by adding textual content
retrieval to complement the existing graph-based knowledge retrieval.
"""

from .service import TextRAGService
from .client import TextRAG
from .models import Document, DocumentChunk, SearchQuery, SearchResult
from .integrations import ContextAwareTextRAG

__all__ = [
    'TextRAGService',
    'TextRAG', 
    'Document',
    'DocumentChunk',
    'SearchQuery',
    'SearchResult',
    'ContextAwareTextRAG'
]