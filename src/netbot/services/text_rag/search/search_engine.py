"""
Main search engine for TextRAG.

Orchestrates semantic search operations including query processing,
vector search, result fusion, and reranking.
"""

import time
from typing import List, Optional, Dict, Any
import numpy as np

from ....shared import get_logger, get_embedding_client
from ....shared.infrastructure.ai import EmbeddingClient
from ..models import SearchQuery, SearchResult, SearchResultSet, SearchMethod
from ..vector_storage.base import VectorStore
from .query_processor import QueryProcessor
from .result_reranker import ResultReranker


class SearchEngine:
    """
    Advanced semantic search engine for TextRAG.
    
    Provides comprehensive search capabilities with multiple strategies,
    result fusion, reranking, and context awareness.
    """
    
    def __init__(self, 
                 vector_store: VectorStore,
                 embedding_client: Optional[EmbeddingClient] = None):
        self.logger = get_logger(__name__)
        self.vector_store = vector_store
        self.embedding_client = embedding_client or get_embedding_client()
        
        # Initialize components
        self.query_processor = QueryProcessor()
        self.result_reranker = ResultReranker()
        
        self.logger.info("Search Engine initialized")
    
    async def search(self, query: SearchQuery) -> SearchResultSet:
        """
        Perform search based on the query configuration.
        
        Args:
            query: Search query with parameters
            
        Returns:
            Complete search result set with metadata
        """
        start_time = time.time()
        
        try:
            # Process the query
            processed_query = await self.query_processor.process_query(query)
            
            # Generate query embedding if needed
            query_embedding = None
            if query.method in [SearchMethod.VECTOR_SIMILARITY, SearchMethod.HYBRID_FUSION, SearchMethod.CONTEXT_AWARE]:
                query_embedding = await self._get_query_embedding(processed_query.text)
            
            # Perform search based on method
            if query.method == SearchMethod.VECTOR_SIMILARITY:
                results = await self._vector_search(query, query_embedding)
            elif query.method == SearchMethod.BM25_LEXICAL:
                results = await self._lexical_search(query)
            elif query.method == SearchMethod.HYBRID_FUSION:
                results = await self._hybrid_search(query, query_embedding)
            elif query.method == SearchMethod.CONTEXT_AWARE:
                results = await self._context_aware_search(query, query_embedding)
            else:
                # Default to hybrid search
                results = await self._hybrid_search(query, query_embedding)
            
            # Apply reranking if enabled
            if query.enable_reranking and len(results) > 1:
                rerank_start = time.time()
                results = await self.result_reranker.rerank_results(
                    results, query.text, query.session_id
                )
                rerank_time = (time.time() - rerank_start) * 1000
            else:
                rerank_time = 0.0
            
            # Apply final filtering
            results = self._apply_final_filters(results, query)
            
            # Create result set
            total_time = (time.time() - start_time) * 1000
            
            result_set = SearchResultSet(
                query_id=query.query_id,
                query_text=query.text,
                search_method=query.method,
                results=results,
                total_results=len(results),
                search_time_ms=total_time,
                reranking_time_ms=rerank_time,
                session_id=query.session_id,
                user_id=query.user_id,
                filters_applied=self._get_applied_filters(query)
            )
            
            self.logger.info(f"Search completed: {len(results)} results in {total_time:.2f}ms")
            return result_set
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            
            # Return empty result set with error info
            return SearchResultSet(
                query_id=query.query_id,
                query_text=query.text,
                search_method=query.method,
                results=[],
                total_results=0,
                search_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _vector_search(self, query: SearchQuery, query_embedding: np.ndarray) -> List[SearchResult]:
        """Perform pure vector similarity search."""
        vector_start = time.time()
        
        try:
            # Build filters
            filters = self._build_filters(query)
            
            # Perform similarity search
            chunk_results = self.vector_store.similarity_search(
                query_embedding=query_embedding,
                top_k=query.top_k,
                similarity_threshold=query.similarity_threshold,
                filters=filters
            )
            
            # Convert to SearchResult objects
            results = []
            for i, (chunk, similarity) in enumerate(chunk_results):
                result = SearchResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    similarity_score=similarity,
                    lexical_score=0.0,
                    fusion_score=similarity,
                    final_rank=i + 1,
                    section_title=chunk.section_title,
                    keywords=chunk.keywords,
                    topics=chunk.topics,
                    named_entities=chunk.named_entities
                )
                results.append(result)
            
            vector_time = (time.time() - vector_start) * 1000
            self.logger.debug(f"Vector search completed in {vector_time:.2f}ms")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return []
    
    async def _lexical_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform pure lexical/BM25 search."""
        lexical_start = time.time()
        
        try:
            # For now, use a simplified lexical search
            # In production, this would use a proper BM25 implementation
            
            # Get all chunks (filtered)
            filters = self._build_filters(query)
            
            # This is a placeholder - in reality we'd need to implement
            # proper BM25 scoring or use a library like Elasticsearch
            query_terms = query.text.lower().split()
            
            # Get documents from vector store (this is not ideal but works for now)
            all_documents = self.vector_store.list_documents()
            
            results = []
            # This would need proper implementation with document retrieval
            # For now, return empty results as this requires more infrastructure
            
            lexical_time = (time.time() - lexical_start) * 1000
            self.logger.debug(f"Lexical search completed in {lexical_time:.2f}ms")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Lexical search failed: {e}")
            return []
    
    async def _hybrid_search(self, query: SearchQuery, query_embedding: np.ndarray) -> List[SearchResult]:
        """Perform hybrid search combining vector and lexical results."""
        hybrid_start = time.time()
        
        try:
            # Build filters
            filters = self._build_filters(query)
            
            # Perform hybrid search using vector store
            results = self.vector_store.hybrid_search(
                query_embedding=query_embedding,
                query_text=query.text,
                top_k=query.top_k,
                vector_weight=query.fusion_weights.get('vector', 0.7),
                lexical_weight=query.fusion_weights.get('lexical', 0.3),
                filters=filters
            )
            
            # Filter by similarity threshold
            filtered_results = [
                r for r in results 
                if r.similarity_score >= query.similarity_threshold
            ]
            
            hybrid_time = (time.time() - hybrid_start) * 1000
            self.logger.debug(f"Hybrid search completed in {hybrid_time:.2f}ms")
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Hybrid search failed: {e}")
            return []
    
    async def _context_aware_search(self, query: SearchQuery, query_embedding: np.ndarray) -> List[SearchResult]:
        """Perform context-aware search using conversation history."""
        try:
            # Start with hybrid search
            results = await self._hybrid_search(query, query_embedding)
            
            # Apply context-aware filtering and boosting
            if query.session_id:
                results = await self._apply_context_boosting(results, query)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Context-aware search failed: {e}")
            return await self._hybrid_search(query, query_embedding)
    
    async def _apply_context_boosting(self, results: List[SearchResult], query: SearchQuery) -> List[SearchResult]:
        """Apply context-based boosting to search results."""
        try:
            # This would integrate with the Context Manager to get conversation history
            # For now, we'll apply simple keyword-based boosting
            
            if not query.conversation_context:
                return results
            
            # Extract keywords from conversation context
            context_keywords = set()
            for context_message in query.conversation_context:
                words = context_message.lower().split()
                context_keywords.update(word for word in words if len(word) > 3)
            
            # Boost results that contain context keywords
            for result in results:
                content_lower = result.content.lower()
                context_boost = 0.0
                
                for keyword in context_keywords:
                    if keyword in content_lower:
                        context_boost += 0.1  # Small boost per matching keyword
                
                # Apply boosting to fusion score
                result.fusion_score = min(1.0, result.fusion_score + context_boost)
            
            # Re-sort by updated fusion scores
            results.sort(key=lambda r: r.fusion_score, reverse=True)
            
            # Update ranks
            for i, result in enumerate(results):
                result.final_rank = i + 1
            
            return results
            
        except Exception as e:
            self.logger.error(f"Context boosting failed: {e}")
            return results
    
    async def _get_query_embedding(self, query_text: str) -> Optional[np.ndarray]:
        """Generate embedding for query text."""
        try:
            embeddings = await self.embedding_client.embed_texts([query_text])
            return embeddings[0] if embeddings else None
        except Exception as e:
            self.logger.error(f"Failed to generate query embedding: {e}")
            return None
    
    def _build_filters(self, query: SearchQuery) -> Dict[str, Any]:
        """Build metadata filters from query parameters."""
        filters = {}
        
        if query.document_ids:
            filters['document_ids'] = query.document_ids
        
        if query.categories:
            filters['categories'] = query.categories
        
        if query.tags:
            filters['tags'] = query.tags
        
        if query.date_from:
            filters['date_from'] = query.date_from
        
        if query.date_to:
            filters['date_to'] = query.date_to
        
        return filters
    
    def _apply_final_filters(self, results: List[SearchResult], query: SearchQuery) -> List[SearchResult]:
        """Apply final filtering to results."""
        # Apply similarity threshold
        filtered = [r for r in results if r.similarity_score >= query.similarity_threshold]
        
        # Apply top_k limit
        filtered = filtered[:query.top_k]
        
        # Add context expansion if requested
        if query.expand_context:
            filtered = self._expand_result_context(filtered)
        
        return filtered
    
    def _expand_result_context(self, results: List[SearchResult]) -> List[SearchResult]:
        """Expand results with adjacent chunk context."""
        # This would fetch adjacent chunks and add them as context
        # For now, just return the original results
        return results
    
    def _get_applied_filters(self, query: SearchQuery) -> List[str]:
        """Get list of filters that were applied."""
        applied = []
        
        if query.document_ids:
            applied.append('document_ids')
        if query.categories:
            applied.append('categories')
        if query.tags:
            applied.append('tags')
        if query.date_from:
            applied.append('date_from')
        if query.date_to:
            applied.append('date_to')
        if query.similarity_threshold > 0.0:
            applied.append('similarity_threshold')
        
        return applied