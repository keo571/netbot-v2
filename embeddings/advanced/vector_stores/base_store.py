"""
Abstract base class for vector stores in the hybrid RAG system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from ..chunking.models import DiagramChunk


class BaseVectorStore(ABC):
    """
    Abstract base class for vector stores.
    
    Defines the interface that all vector store implementations must follow
    for the hybrid RAG system.
    """
    
    @abstractmethod
    def add_chunks(self, chunks: List[DiagramChunk]) -> bool:
        """
        Add chunks to the vector store.
        
        Args:
            chunks: List of DiagramChunk objects with embeddings
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def search(self, 
               query_embedding: List[float], 
               top_k: int = 10,
               filters: Optional[Dict[str, Any]] = None) -> List[Tuple[DiagramChunk, float]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query vector embedding
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        pass
    
    @abstractmethod
    def get_chunk_by_id(self, chunk_id: str) -> Optional[DiagramChunk]:
        """
        Retrieve a specific chunk by ID.
        
        Args:
            chunk_id: Unique chunk identifier
            
        Returns:
            DiagramChunk if found, None otherwise
        """
        pass
    
    @abstractmethod
    def delete_chunks(self, chunk_ids: List[str]) -> bool:
        """
        Delete chunks from the vector store.
        
        Args:
            chunk_ids: List of chunk IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def list_chunks_by_diagram(self, diagram_id: str) -> List[DiagramChunk]:
        """
        Get all chunks that reference a specific diagram.
        
        Args:
            diagram_id: Diagram identifier
            
        Returns:
            List of chunks referencing the diagram
        """
        pass
    
    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with stats (e.g., total_chunks, diagrams_count, etc.)
        """
        pass
    
    @abstractmethod
    def clear_collection(self) -> bool:
        """
        Clear all data from the vector store.
        
        Returns:
            True if successful, False otherwise
        """
        pass