"""
Base vector storage interface for TextRAG.

Defines the abstract interface for vector storage backends
supporting embedding storage and similarity search.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from ..models import DocumentChunk, SearchQuery, SearchResult


class VectorStore(ABC):
    """
    Abstract base class for vector storage backends.
    
    Defines the interface for storing and searching document embeddings
    across different vector database implementations.
    """
    
    @abstractmethod
    def initialize(self, collection_name: str = "textrag_chunks") -> None:
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the collection to use
        """
        pass
    
    @abstractmethod
    def add_chunk(self, chunk: DocumentChunk) -> bool:
        """
        Add a document chunk with its embedding to the store.
        
        Args:
            chunk: Document chunk with embedding
            
        Returns:
            True if added successfully
        """
        pass
    
    @abstractmethod
    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Add multiple document chunks to the store.
        
        Args:
            chunks: List of document chunks with embeddings
            
        Returns:
            Number of chunks added successfully
        """
        pass
    
    @abstractmethod
    def update_chunk(self, chunk: DocumentChunk) -> bool:
        """
        Update an existing chunk in the store.
        
        Args:
            chunk: Updated document chunk
            
        Returns:
            True if updated successfully
        """
        pass
    
    @abstractmethod
    def delete_chunk(self, chunk_id: str) -> bool:
        """
        Delete a chunk from the store.
        
        Args:
            chunk_id: ID of chunk to delete
            
        Returns:
            True if deleted successfully
        """
        pass
    
    @abstractmethod
    def delete_document_chunks(self, document_id: str) -> int:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            Number of chunks deleted
        """
        pass
    
    @abstractmethod
    def similarity_search(self, 
                         query_embedding: np.ndarray,
                         top_k: int = 5,
                         similarity_threshold: float = 0.7,
                         filters: Optional[Dict[str, Any]] = None) -> List[Tuple[DocumentChunk, float]]:
        """
        Perform similarity search using query embedding.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            filters: Optional metadata filters
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        pass
    
    @abstractmethod
    def hybrid_search(self, 
                     query_embedding: np.ndarray,
                     query_text: str,
                     top_k: int = 5,
                     vector_weight: float = 0.7,
                     lexical_weight: float = 0.3,
                     filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Perform hybrid search combining vector and lexical search.
        
        Args:
            query_embedding: Query vector
            query_text: Query text for lexical search
            top_k: Number of results to return
            vector_weight: Weight for vector search
            lexical_weight: Weight for lexical search
            filters: Optional metadata filters
            
        Returns:
            List of search results with combined scores
        """
        pass
    
    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """
        Retrieve a chunk by ID.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            DocumentChunk if found, None otherwise
        """
        pass
    
    @abstractmethod
    def list_documents(self) -> List[str]:
        """
        List all document IDs in the store.
        
        Returns:
            List of document IDs
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with store statistics
        """
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """
        Clear all data from the store.
        
        Returns:
            True if cleared successfully
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Clean up and close connections."""
        pass
    
    # Helper methods that can be overridden
    
    def calculate_similarity(self, 
                           embedding1: np.ndarray, 
                           embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score (0-1)
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # Ensure result is in [0, 1] range
        return max(0.0, min(1.0, (similarity + 1.0) / 2.0))
    
    def lexical_search(self, 
                      query_text: str,
                      chunks: List[DocumentChunk],
                      top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """
        Simple lexical search using term frequency.
        
        Args:
            query_text: Query text
            chunks: List of chunks to search
            top_k: Number of results to return
            
        Returns:
            List of (chunk, score) tuples
        """
        query_terms = query_text.lower().split()
        if not query_terms:
            return []
        
        chunk_scores = []
        
        for chunk in chunks:
            content_lower = chunk.content.lower()
            
            # Calculate simple TF score
            score = 0.0
            for term in query_terms:
                tf = content_lower.count(term) / len(content_lower.split()) if content_lower else 0
                score += tf
            
            if score > 0:
                chunk_scores.append((chunk, score))
        
        # Sort by score and return top_k
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        return chunk_scores[:top_k]