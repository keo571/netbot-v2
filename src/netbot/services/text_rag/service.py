"""
Main TextRAG service for NetBot V2.

Orchestrates document processing, vector storage, and semantic search
to provide comprehensive text-based retrieval capabilities.
"""

import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

from ...shared import get_logger, get_embedding_client
from ...shared.infrastructure.ai import EmbeddingClient
from .models import (
    Document, DocumentChunk, DocumentStatus, SearchQuery, 
    SearchResultSet, ChunkingStrategy
)
from .repository import TextRAGRepository
from .processing import DocumentProcessor, TextChunker, ChunkConfig
from .vector_storage import VectorStore, InMemoryVectorStore, ChromaVectorStore
from .search import SearchEngine


class TextRAGService:
    """
    Main TextRAG service.
    
    Provides comprehensive text processing, storage, and retrieval
    capabilities for the NetBot V2 hybrid RAG architecture.
    """
    
    def __init__(self, 
                 repository: Optional[TextRAGRepository] = None,
                 vector_store: Optional[VectorStore] = None,
                 embedding_client: Optional[EmbeddingClient] = None):
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.repository = repository or TextRAGRepository()
        self.embedding_client = embedding_client or get_embedding_client()
        
        # Initialize vector store
        if vector_store is None:
            try:
                # Try to use ChromaDB if available
                self.vector_store = ChromaVectorStore()
                self.vector_store.initialize()
            except ImportError:
                # Fall back to in-memory store
                self.logger.warning("ChromaDB not available, using in-memory vector store")
                self.vector_store = InMemoryVectorStore()
                self.vector_store.initialize()
        else:
            self.vector_store = vector_store
        
        # Initialize processing and search components
        self.document_processor = DocumentProcessor(self.repository)
        self.search_engine = SearchEngine(self.vector_store, self.embedding_client)
        
        self.logger.info("TextRAG Service initialized")
    
    # === Document Management ===
    
    async def add_document_from_file(self, 
                                   file_path: Union[str, Path],
                                   title: Optional[str] = None,
                                   categories: Optional[List[str]] = None,
                                   tags: Optional[List[str]] = None,
                                   chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE_CHARACTER,
                                   chunk_size: int = 1000,
                                   chunk_overlap: int = 200) -> Document:
        """
        Add a document from a file.
        
        Args:
            file_path: Path to the document file
            title: Document title (defaults to filename)
            categories: Document categories
            tags: Document tags
            chunking_strategy: Text chunking strategy
            chunk_size: Target chunk size
            chunk_overlap: Chunk overlap
            
        Returns:
            Processed document
        """
        chunk_config = ChunkConfig(
            strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Process the document
        document = await self.document_processor.process_file(
            file_path=file_path,
            title=title,
            categories=categories,
            tags=tags,
            chunk_config=chunk_config
        )
        
        # Generate embeddings for chunks and add to vector store
        await self._process_document_embeddings(document.document_id)
        
        return document
    
    async def add_document_from_content(self, 
                                      content: str,
                                      title: str,
                                      document_type: str,
                                      source_url: Optional[str] = None,
                                      author: Optional[str] = None,
                                      categories: Optional[List[str]] = None,
                                      tags: Optional[List[str]] = None,
                                      chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE_CHARACTER,
                                      chunk_size: int = 1000,
                                      chunk_overlap: int = 200) -> Document:
        """
        Add a document from text content.
        
        Args:
            content: Document text content
            title: Document title
            document_type: Type of document
            source_url: Source URL if applicable
            author: Document author
            categories: Document categories
            tags: Document tags
            chunking_strategy: Text chunking strategy
            chunk_size: Target chunk size
            chunk_overlap: Chunk overlap
            
        Returns:
            Processed document
        """
        from .models import DocumentType
        
        chunk_config = ChunkConfig(
            strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Process the document
        document = await self.document_processor.process_content(
            content=content,
            title=title,
            document_type=DocumentType(document_type),
            source_url=source_url,
            author=author,
            categories=categories,
            tags=tags,
            chunk_config=chunk_config
        )
        
        # Generate embeddings for chunks and add to vector store
        await self._process_document_embeddings(document.document_id)
        
        return document
    
    async def update_document(self, 
                            document_id: str,
                            chunking_strategy: Optional[ChunkingStrategy] = None,
                            chunk_size: Optional[int] = None,
                            chunk_overlap: Optional[int] = None) -> Optional[Document]:
        """
        Update a document with new processing parameters.
        
        Args:
            document_id: Document to update
            chunking_strategy: New chunking strategy
            chunk_size: New chunk size
            chunk_overlap: New chunk overlap
            
        Returns:
            Updated document or None if not found
        """
        # Get existing document
        document = self.repository.get_document(document_id)
        if not document:
            return None
        
        # Create chunk config with updates
        chunk_config = ChunkConfig(
            strategy=chunking_strategy or document.chunking_strategy,
            chunk_size=chunk_size or document.chunk_size,
            chunk_overlap=chunk_overlap or document.chunk_overlap
        )
        
        # Remove old chunks from vector store
        self.vector_store.delete_document_chunks(document_id)
        
        # Reprocess document
        document = await self.document_processor.reprocess_document(
            document_id, chunk_config
        )
        
        if document:
            # Generate new embeddings
            await self._process_document_embeddings(document_id)
        
        return document
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks.
        
        Args:
            document_id: Document to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            # Remove from vector store
            self.vector_store.delete_document_chunks(document_id)
            
            # Remove from database
            return self.repository.delete_document(document_id)
            
        except Exception as e:
            self.logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self.repository.get_document(document_id)
    
    def list_documents(self, 
                      status: Optional[DocumentStatus] = None,
                      categories: Optional[List[str]] = None,
                      limit: int = 100) -> List[Document]:
        """List documents with optional filtering."""
        return self.repository.list_documents(status, categories, limit)
    
    # === Search Operations ===
    
    async def search(self, 
                    query_text: str,
                    top_k: int = 5,
                    similarity_threshold: float = 0.7,
                    document_ids: Optional[List[str]] = None,
                    categories: Optional[List[str]] = None,
                    session_id: Optional[str] = None,
                    user_id: Optional[str] = None) -> SearchResultSet:
        """
        Perform semantic search across documents.
        
        Args:
            query_text: Search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            document_ids: Limit search to specific documents
            categories: Filter by categories
            session_id: Session ID for context-aware search
            user_id: User ID for personalized search
            
        Returns:
            Search results with metadata
        """
        # Create search query
        search_query = SearchQuery(
            text=query_text,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            document_ids=document_ids,
            categories=categories,
            session_id=session_id,
            user_id=user_id
        )
        
        # Perform search
        result_set = await self.search_engine.search(search_query)
        
        # Record search for analytics
        self.repository.record_search(result_set)
        
        return result_set
    
    async def context_aware_search(self,
                                 query_text: str,
                                 session_id: str,
                                 conversation_context: Optional[List[str]] = None,
                                 top_k: int = 5,
                                 similarity_threshold: float = 0.7) -> SearchResultSet:
        """
        Perform context-aware search using conversation history.
        
        Args:
            query_text: Search query
            session_id: Session ID for context
            conversation_context: Recent conversation messages
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            Context-aware search results
        """
        from .models import SearchMethod
        
        search_query = SearchQuery(
            text=query_text,
            method=SearchMethod.CONTEXT_AWARE,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            session_id=session_id,
            conversation_context=conversation_context
        )
        
        result_set = await self.search_engine.search(search_query)
        self.repository.record_search(result_set)
        
        return result_set
    
    # === Batch Operations ===
    
    async def batch_add_documents(self, 
                                 file_paths: List[Union[str, Path]],
                                 categories: Optional[List[str]] = None,
                                 tags: Optional[List[str]] = None,
                                 max_concurrent: int = 5) -> List[Document]:
        """
        Process multiple documents concurrently.
        
        Args:
            file_paths: List of file paths to process
            categories: Default categories for all documents
            tags: Default tags for all documents
            max_concurrent: Maximum concurrent processing
            
        Returns:
            List of processed documents
        """
        chunk_config = ChunkConfig(
            strategy=ChunkingStrategy.RECURSIVE_CHARACTER,
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Process documents
        documents = await self.document_processor.batch_process(
            file_paths=file_paths,
            categories=categories,
            tags=tags,
            chunk_config=chunk_config,
            max_concurrent=max_concurrent
        )
        
        # Generate embeddings for all documents
        embedding_tasks = [
            self._process_document_embeddings(doc.document_id) 
            for doc in documents
        ]
        
        await asyncio.gather(*embedding_tasks, return_exceptions=True)
        
        return documents
    
    # === Analytics and Maintenance ===
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        doc_stats = self.repository.get_document_statistics()
        vector_stats = self.vector_store.get_stats()
        search_stats = self.repository.get_search_statistics()
        
        return {
            'service': 'TextRAG',
            'documents': doc_stats,
            'vector_storage': vector_stats,
            'search': search_stats
        }
    
    async def rebuild_embeddings(self, document_id: Optional[str] = None) -> int:
        """
        Rebuild embeddings for documents.
        
        Args:
            document_id: Specific document to rebuild (or all if None)
            
        Returns:
            Number of documents processed
        """
        if document_id:
            # Rebuild for specific document
            await self._process_document_embeddings(document_id)
            return 1
        else:
            # Rebuild for all documents
            documents = self.repository.list_documents(status=DocumentStatus.COMPLETED)
            
            for document in documents:
                try:
                    await self._process_document_embeddings(document.document_id)
                except Exception as e:
                    self.logger.error(f"Failed to rebuild embeddings for {document.document_id}: {e}")
            
            return len(documents)
    
    def cleanup_failed_documents(self) -> int:
        """Remove documents that failed processing."""
        failed_docs = self.repository.list_documents(status=DocumentStatus.FAILED)
        
        cleaned_count = 0
        for doc in failed_docs:
            if self.repository.delete_document(doc.document_id):
                cleaned_count += 1
        
        return cleaned_count
    
    # === Private Methods ===
    
    async def _process_document_embeddings(self, document_id: str) -> None:
        """Generate and store embeddings for document chunks."""
        try:
            # Get document chunks
            chunks = self.repository.get_document_chunks(document_id)
            
            if not chunks:
                self.logger.warning(f"No chunks found for document: {document_id}")
                return
            
            # Generate embeddings in batches
            batch_size = 50
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                
                # Extract text content
                texts = [chunk.content for chunk in batch_chunks]
                
                # Generate embeddings
                embeddings = await self.embedding_client.embed_texts(texts)
                
                # Update chunks with embeddings
                for chunk, embedding in zip(batch_chunks, embeddings):
                    chunk.embedding = embedding
                    chunk.embedding_model = self.embedding_client.model_name
                    
                    # Update in repository
                    self.repository.update_chunk_embedding(
                        chunk.chunk_id, 
                        embedding, 
                        self.embedding_client.model_name
                    )
                
                # Add to vector store
                self.vector_store.add_chunks(batch_chunks)
            
            self.logger.info(f"Generated embeddings for {len(chunks)} chunks in document {document_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to process embeddings for document {document_id}: {e}")
            raise
    
    def close(self) -> None:
        """Clean up resources."""
        try:
            self.vector_store.close()
            self.logger.info("TextRAG Service closed")
        except Exception as e:
            self.logger.error(f"Error closing TextRAG Service: {e}")