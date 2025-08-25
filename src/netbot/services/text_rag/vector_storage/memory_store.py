"""
In-memory vector storage for TextRAG.

Provides a simple in-memory vector store for development and testing.
Uses numpy for similarity calculations and basic indexing.
"""

import threading
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from ....shared import get_logger
from .base import VectorStore
from ..models import DocumentChunk, SearchResult


class InMemoryVectorStore(VectorStore):
    """
    In-memory vector storage implementation.
    
    Provides fast, ephemeral vector storage using numpy arrays
    and basic similarity search algorithms.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._lock = threading.RLock()
        
        # Storage
        self._chunks: Dict[str, DocumentChunk] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        self._document_index: Dict[str, List[str]] = {}  # document_id -> chunk_ids
        
        self.collection_name = "textrag_chunks"
        self.initialized = False
    
    def initialize(self, collection_name: str = "textrag_chunks") -> None:
        """Initialize the in-memory store."""
        with self._lock:
            self.collection_name = collection_name
            self.initialized = True
            self.logger.info(f"Initialized InMemoryVectorStore collection: {collection_name}")
    
    def add_chunk(self, chunk: DocumentChunk) -> bool:
        """Add a single chunk to the store."""
        if not self.initialized:
            self.initialize()
        
        with self._lock:
            try:
                # Store chunk
                self._chunks[chunk.chunk_id] = chunk
                
                # Store embedding if available
                if chunk.embedding is not None:
                    self._embeddings[chunk.chunk_id] = chunk.embedding.copy()
                
                # Update document index
                if chunk.document_id not in self._document_index:
                    self._document_index[chunk.document_id] = []
                self._document_index[chunk.document_id].append(chunk.chunk_id)
                
                self.logger.debug(f"Added chunk: {chunk.chunk_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to add chunk {chunk.chunk_id}: {e}")
                return False
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Add multiple chunks to the store."""
        if not self.initialized:
            self.initialize()
        
        added_count = 0
        for chunk in chunks:
            if self.add_chunk(chunk):
                added_count += 1
        
        self.logger.info(f"Added {added_count} out of {len(chunks)} chunks")
        return added_count
    
    def update_chunk(self, chunk: DocumentChunk) -> bool:
        """Update an existing chunk."""
        with self._lock:
            if chunk.chunk_id not in self._chunks:
                self.logger.warning(f"Chunk not found for update: {chunk.chunk_id}")
                return False
            
            try:
                # Update chunk
                self._chunks[chunk.chunk_id] = chunk
                
                # Update embedding if available
                if chunk.embedding is not None:
                    self._embeddings[chunk.chunk_id] = chunk.embedding.copy()
                
                self.logger.debug(f"Updated chunk: {chunk.chunk_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to update chunk {chunk.chunk_id}: {e}")
                return False
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk from the store."""
        with self._lock:
            try:
                if chunk_id not in self._chunks:
                    return False
                
                # Get chunk to find document_id
                chunk = self._chunks[chunk_id]
                
                # Remove from storage
                del self._chunks[chunk_id]
                
                # Remove embedding
                if chunk_id in self._embeddings:
                    del self._embeddings[chunk_id]
                
                # Update document index
                if chunk.document_id in self._document_index:
                    if chunk_id in self._document_index[chunk.document_id]:
                        self._document_index[chunk.document_id].remove(chunk_id)
                    
                    # Clean up empty document entries
                    if not self._document_index[chunk.document_id]:
                        del self._document_index[chunk.document_id]
                
                self.logger.debug(f"Deleted chunk: {chunk_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to delete chunk {chunk_id}: {e}")
                return False
    
    def delete_document_chunks(self, document_id: str) -> int:
        """Delete all chunks for a document."""
        with self._lock:
            if document_id not in self._document_index:
                return 0
            
            chunk_ids = self._document_index[document_id].copy()
            deleted_count = 0
            
            for chunk_id in chunk_ids:
                if self.delete_chunk(chunk_id):
                    deleted_count += 1
            
            self.logger.info(f"Deleted {deleted_count} chunks for document: {document_id}")
            return deleted_count
    
    def similarity_search(self, 
                         query_embedding: np.ndarray,
                         top_k: int = 5,
                         similarity_threshold: float = 0.7,
                         filters: Optional[Dict[str, Any]] = None) -> List[Tuple[DocumentChunk, float]]:
        """Perform similarity search using query embedding."""
        with self._lock:
            if query_embedding is None:
                return []
            
            results = []
            
            for chunk_id, chunk in self._chunks.items():
                # Skip chunks without embeddings
                if chunk_id not in self._embeddings:
                    continue
                
                # Apply filters
                if filters and not self._apply_filters(chunk, filters):
                    continue
                
                # Calculate similarity
                embedding = self._embeddings[chunk_id]
                similarity = self.calculate_similarity(query_embedding, embedding)
                
                # Check threshold
                if similarity >= similarity_threshold:
                    results.append((chunk, similarity))
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
    
    def hybrid_search(self, 
                     query_embedding: np.ndarray,
                     query_text: str,
                     top_k: int = 5,
                     vector_weight: float = 0.7,
                     lexical_weight: float = 0.3,
                     filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Perform hybrid search combining vector and lexical search."""
        with self._lock:
            # Get vector search results
            vector_results = self.similarity_search(
                query_embedding, 
                top_k=top_k*2,  # Get more for fusion
                similarity_threshold=0.0,
                filters=filters
            )
            
            # Get all chunks for lexical search
            all_chunks = [chunk for chunk in self._chunks.values() 
                         if not filters or self._apply_filters(chunk, filters)]
            
            # Get lexical search results
            lexical_results = self.lexical_search(query_text, all_chunks, top_k=top_k*2)
            
            # Combine and score results
            combined_results = {}
            
            # Add vector results
            for chunk, vector_score in vector_results:
                combined_results[chunk.chunk_id] = {
                    'chunk': chunk,
                    'vector_score': vector_score,
                    'lexical_score': 0.0
                }
            
            # Add lexical results
            for chunk, lexical_score in lexical_results:
                if chunk.chunk_id in combined_results:
                    combined_results[chunk.chunk_id]['lexical_score'] = lexical_score
                else:
                    combined_results[chunk.chunk_id] = {
                        'chunk': chunk,
                        'vector_score': 0.0,
                        'lexical_score': lexical_score
                    }
            
            # Calculate fusion scores and create SearchResult objects
            search_results = []
            for result_data in combined_results.values():
                chunk = result_data['chunk']
                vector_score = result_data['vector_score']
                lexical_score = result_data['lexical_score']
                fusion_score = (vector_weight * vector_score) + (lexical_weight * lexical_score)
                
                search_result = SearchResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    similarity_score=vector_score,
                    lexical_score=lexical_score,
                    fusion_score=fusion_score,
                    section_title=chunk.section_title,
                    keywords=chunk.keywords,
                    topics=chunk.topics,
                    named_entities=chunk.named_entities
                )
                
                search_results.append(search_result)
            
            # Sort by fusion score and return top_k
            search_results.sort(key=lambda x: x.fusion_score, reverse=True)
            
            # Set final ranks
            for i, result in enumerate(search_results[:top_k]):
                result.final_rank = i + 1
            
            return search_results[:top_k]
    
    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Retrieve a chunk by ID."""
        with self._lock:
            return self._chunks.get(chunk_id)
    
    def list_documents(self) -> List[str]:
        """List all document IDs in the store."""
        with self._lock:
            return list(self._document_index.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        with self._lock:
            total_chunks = len(self._chunks)
            chunks_with_embeddings = len(self._embeddings)
            total_documents = len(self._document_index)
            
            # Calculate average chunks per document
            avg_chunks_per_doc = total_chunks / total_documents if total_documents > 0 else 0
            
            # Get embedding dimensions if we have any
            embedding_dim = None
            if self._embeddings:
                first_embedding = next(iter(self._embeddings.values()))
                embedding_dim = len(first_embedding)
            
            return {
                'backend': 'memory',
                'collection_name': self.collection_name,
                'total_chunks': total_chunks,
                'chunks_with_embeddings': chunks_with_embeddings,
                'total_documents': total_documents,
                'avg_chunks_per_document': avg_chunks_per_doc,
                'embedding_dimensions': embedding_dim,
                'memory_usage': 'not_tracked'
            }
    
    def clear(self) -> bool:
        """Clear all data from the store."""
        with self._lock:
            try:
                self._chunks.clear()
                self._embeddings.clear()
                self._document_index.clear()
                
                self.logger.info("Cleared all data from InMemoryVectorStore")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to clear store: {e}")
                return False
    
    def close(self) -> None:
        """Clean up and close connections."""
        with self._lock:
            self.clear()
            self.initialized = False
            self.logger.info("Closed InMemoryVectorStore")
    
    def _apply_filters(self, chunk: DocumentChunk, filters: Dict[str, Any]) -> bool:
        """Apply metadata filters to a chunk."""
        try:
            # Document ID filter
            if 'document_ids' in filters:
                if chunk.document_id not in filters['document_ids']:
                    return False
            
            # Category filter (would need to get from document)
            if 'categories' in filters:
                # This would require loading the document to get categories
                # For now, we'll skip this filter
                pass
            
            # Date range filter
            if 'date_from' in filters:
                if chunk.created_at < filters['date_from']:
                    return False
            
            if 'date_to' in filters:
                if chunk.created_at > filters['date_to']:
                    return False
            
            # Keywords filter
            if 'keywords' in filters:
                chunk_keywords_lower = [kw.lower() for kw in chunk.keywords]
                filter_keywords_lower = [kw.lower() for kw in filters['keywords']]
                
                if not any(kw in chunk_keywords_lower for kw in filter_keywords_lower):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying filters: {e}")
            return True  # Default to including the chunk