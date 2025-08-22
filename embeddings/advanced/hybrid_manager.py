"""
Enhanced embedding manager for hybrid RAG system.

Manages both diagram + context chunks and graph node embeddings,
providing the foundation for hybrid retrieval.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .chunking import HybridChunker, DiagramChunk, ChunkType
from .vector_stores import BaseVectorStore, ChromaVectorStore
from ..embedding_encoder import EmbeddingEncoder


logger = logging.getLogger(__name__)


class HybridEmbeddingManager:
    """
    Enhanced embedding manager for hybrid RAG system.
    
    Combines:
    1. Document chunking with diagram + context approach
    2. Vector storage for semantic search
    3. Integration with graph embeddings
    4. Hybrid retrieval orchestration
    """
    
    def __init__(self,
                 vector_store: Optional[BaseVectorStore] = None,
                 chunker: Optional[HybridChunker] = None,
                 embedding_encoder: Optional[EmbeddingEncoder] = None):
        """
        Initialize the hybrid embedding manager.
        
        Args:
            vector_store: Vector database for chunk storage
            chunker: Strategy for creating diagram + context chunks
            embedding_encoder: Service for computing embeddings
        """
        # Initialize components
        self.vector_store = vector_store or ChromaVectorStore()
        self.chunker = chunker or HybridChunker()
        self.embedding_encoder = embedding_encoder or EmbeddingEncoder()
        
        logger.info("Initialized HybridEmbeddingManager")
    
    def process_document(self, 
                        document_text: str,
                        source_document: str,
                        known_diagram_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process a document into chunks and store embeddings.
        
        This is the core method for your hybrid RAG approach:
        1. Chunks document text with diagram + context strategy
        2. Computes embeddings for each chunk
        3. Stores chunks in vector database
        
        Args:
            document_text: Full text of the document
            source_document: Name/path of source document
            known_diagram_ids: List of diagram IDs that exist in the document
            
        Returns:
            Dictionary with processing results and statistics
        """
        logger.info(f"Processing document: {source_document}")
        
        try:
            # Step 1: Create chunks using hybrid strategy
            chunks = self.chunker.chunk_document(
                text=document_text,
                source_document=source_document,
                known_diagram_ids=known_diagram_ids or []
            )
            
            logger.info(f"Created {len(chunks)} chunks")
            
            # Step 2: Compute embeddings for chunks
            chunks_with_embeddings = self._compute_chunk_embeddings(chunks)
            
            # Step 3: Store chunks in vector database
            success = self.vector_store.add_chunks(chunks_with_embeddings)
            
            if not success:
                raise Exception("Failed to store chunks in vector database")
            
            # Step 4: Analyze results
            stats = self._analyze_chunks(chunks_with_embeddings)
            
            result = {
                "status": "success",
                "source_document": source_document,
                "total_chunks": len(chunks_with_embeddings),
                "chunks_with_diagrams": stats["chunks_with_diagrams"],
                "unique_diagrams": stats["unique_diagrams"], 
                "diagram_ids": stats["diagram_ids"],
                "chunk_types": stats["chunk_types"]
            }
            
            logger.info(f"Successfully processed document: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {source_document}: {e}")
            return {
                "status": "error",
                "source_document": source_document,
                "error": str(e)
            }
    
    def search_hybrid(self, 
                     query: str,
                     top_k: int = 10,
                     diagram_only: bool = False) -> List[Tuple[DiagramChunk, float]]:
        """
        Search for relevant chunks using hybrid approach.
        
        This is the core search method for your hybrid RAG workflow:
        1. Computes query embedding
        2. Searches vector database for similar chunks
        3. Returns chunks that can trigger graph retrieval
        
        Args:
            query: Search query text
            top_k: Number of results to return
            diagram_only: If True, only return chunks with diagram references
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        logger.info(f"Hybrid search query: '{query}' (top_k={top_k})")
        
        try:
            # Step 1: Compute query embedding
            query_embedding = self.embedding_encoder.encode_text(query)
            
            # Step 2: Search vector database
            filters = {}
            if diagram_only:
                filters["has_diagram"] = True
            
            results = self.vector_store.search(
                query_embedding=query_embedding.tolist(),
                top_k=top_k,
                filters=filters
            )
            
            logger.info(f"Found {len(results)} relevant chunks")
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    def get_diagram_chunks(self, diagram_id: str) -> List[DiagramChunk]:
        """
        Get all chunks that reference a specific diagram.
        
        Useful for understanding what context is associated with a diagram.
        """
        return self.vector_store.list_chunks_by_diagram(diagram_id)
    
    def extract_diagram_ids_from_results(self, 
                                       search_results: List[Tuple[DiagramChunk, float]]) -> List[str]:
        """
        Extract diagram IDs from search results.
        
        This enables the hybrid workflow: vector search finds relevant chunks,
        then graph search is triggered for the associated diagrams.
        """
        diagram_ids = []
        for chunk, score in search_results:
            if chunk.has_diagram_reference():
                diagram_ids.append(chunk.diagram_id)
                
                # Also check for additional diagram IDs in properties
                if chunk.properties and "all_diagram_ids" in chunk.properties:
                    additional_ids = chunk.properties["all_diagram_ids"]
                    if isinstance(additional_ids, list):
                        diagram_ids.extend(additional_ids)
        
        # Return unique diagram IDs
        return list(set(diagram_ids))
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the embedding collection."""
        return self.vector_store.get_collection_stats()
    
    def clear_all_embeddings(self) -> bool:
        """Clear all stored embeddings (use with caution)."""
        logger.warning("Clearing all embeddings")
        return self.vector_store.clear_collection()
    
    def _compute_chunk_embeddings(self, chunks: List[DiagramChunk]) -> List[DiagramChunk]:
        """Compute embeddings for a list of chunks."""
        logger.info(f"Computing embeddings for {len(chunks)} chunks")
        
        # Prepare texts for batch embedding
        texts = [chunk.get_full_content() for chunk in chunks]
        
        # Compute embeddings in batch (much more efficient)
        embeddings = self.embedding_encoder.batch_encode_texts(texts)
        
        # Assign embeddings to chunks
        chunks_with_embeddings = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding.tolist()
            chunks_with_embeddings.append(chunk)
        
        return chunks_with_embeddings
    
    def _analyze_chunks(self, chunks: List[DiagramChunk]) -> Dict[str, Any]:
        """Analyze chunk statistics."""
        chunks_with_diagrams = 0
        diagram_ids = set()
        chunk_types = {}
        
        for chunk in chunks:
            if chunk.has_diagram_reference():
                chunks_with_diagrams += 1
                diagram_ids.add(chunk.diagram_id)
            
            chunk_type = chunk.chunk_type.value
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        return {
            "chunks_with_diagrams": chunks_with_diagrams,
            "unique_diagrams": len(diagram_ids),
            "diagram_ids": list(diagram_ids),
            "chunk_types": chunk_types
        }