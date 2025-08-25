"""
TextRAG client interface for NetBot V2.

Provides a simple, clean interface for integrating text-based retrieval
capabilities into applications and other services.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any, Union

from ...shared import get_logger
from .service import TextRAGService
from .models import Document, SearchResultSet, DocumentStatus, ChunkingStrategy
from .vector_storage import VectorStore, InMemoryVectorStore


class TextRAG:
    """
    Main client interface for TextRAG functionality.
    
    Provides a simplified API for document processing, storage, and semantic search
    that integrates seamlessly with the NetBot V2 hybrid RAG architecture.
    """
    
    def __init__(self, 
                 vector_store: Optional[VectorStore] = None,
                 use_persistent_storage: bool = True):
        """
        Initialize TextRAG client.
        
        Args:
            vector_store: Custom vector store backend
            use_persistent_storage: Use persistent storage (ChromaDB) if available
        """
        self.logger = get_logger(__name__)
        
        # Initialize service with appropriate vector store
        if vector_store is None and not use_persistent_storage:
            vector_store = InMemoryVectorStore()
            vector_store.initialize()
        
        self.service = TextRAGService(vector_store=vector_store)
        self.logger.info("TextRAG client initialized")
    
    # === Document Operations ===
    
    async def add_document(self, 
                          file_path: Union[str, Path],
                          title: Optional[str] = None,
                          categories: Optional[List[str]] = None,
                          tags: Optional[List[str]] = None) -> Document:
        """
        Add a document from a file.
        
        Args:
            file_path: Path to the document file
            title: Document title (defaults to filename)
            categories: Document categories for organization
            tags: Document tags for filtering
            
        Returns:
            Processed document with chunks and embeddings
            
        Example:
            >>> textrag = TextRAG()
            >>> doc = await textrag.add_document("data/manual.pdf", 
            ...                                  categories=["documentation"])
            >>> print(f"Added document: {doc.title} with {len(doc.chunks)} chunks")
        """
        return await self.service.add_document_from_file(
            file_path=file_path,
            title=title,
            categories=categories,
            tags=tags
        )
    
    async def add_text_document(self, 
                               content: str,
                               title: str,
                               categories: Optional[List[str]] = None,
                               tags: Optional[List[str]] = None,
                               source_url: Optional[str] = None) -> Document:
        """
        Add a document from text content.
        
        Args:
            content: Document text content
            title: Document title
            categories: Document categories
            tags: Document tags
            source_url: Source URL if applicable
            
        Returns:
            Processed document
            
        Example:
            >>> content = "This is a network configuration guide..."
            >>> doc = await textrag.add_text_document(
            ...     content=content,
            ...     title="Network Config Guide",
            ...     categories=["guides"]
            ... )
        """
        return await self.service.add_document_from_content(
            content=content,
            title=title,
            document_type="text",
            source_url=source_url,
            categories=categories,
            tags=tags
        )
    
    async def add_web_document(self, 
                              url: str,
                              title: Optional[str] = None,
                              categories: Optional[List[str]] = None,
                              tags: Optional[List[str]] = None) -> Optional[Document]:
        """
        Add a document from a web URL.
        
        Args:
            url: URL to fetch content from
            title: Document title (defaults to page title)
            categories: Document categories
            tags: Document tags
            
        Returns:
            Processed document or None if failed
            
        Example:
            >>> doc = await textrag.add_web_document(
            ...     "https://docs.example.com/api",
            ...     categories=["api-docs"]
            ... )
        """
        try:
            # This would use a web scraper to fetch content
            # For now, this is a placeholder implementation
            self.logger.warning("Web document fetching not implemented")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to add web document from {url}: {e}")
            return None
    
    async def batch_add_documents(self, 
                                 file_paths: List[Union[str, Path]],
                                 categories: Optional[List[str]] = None,
                                 tags: Optional[List[str]] = None) -> List[Document]:
        """
        Add multiple documents at once.
        
        Args:
            file_paths: List of file paths to process
            categories: Default categories for all documents
            tags: Default tags for all documents
            
        Returns:
            List of processed documents
            
        Example:
            >>> docs = await textrag.batch_add_documents([
            ...     "docs/guide1.pdf",
            ...     "docs/guide2.md",
            ...     "docs/api.json"
            ... ], categories=["documentation"])
        """
        return await self.service.batch_add_documents(
            file_paths=file_paths,
            categories=categories,
            tags=tags
        )
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """
        Get a document by ID.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document if found, None otherwise
        """
        return self.service.get_document(document_id)
    
    def list_documents(self, 
                      categories: Optional[List[str]] = None,
                      status: Optional[str] = None,
                      limit: int = 100) -> List[Document]:
        """
        List documents with optional filtering.
        
        Args:
            categories: Filter by categories
            status: Filter by processing status
            limit: Maximum number of documents
            
        Returns:
            List of matching documents
        """
        status_filter = None
        if status:
            status_filter = DocumentStatus(status)
        
        return self.service.list_documents(
            status=status_filter,
            categories=categories,
            limit=limit
        )
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks.
        
        Args:
            document_id: Document to delete
            
        Returns:
            True if deleted successfully
        """
        return await self.service.delete_document(document_id)
    
    # === Search Operations ===
    
    async def search(self, 
                    query: str,
                    top_k: int = 5,
                    categories: Optional[List[str]] = None,
                    similarity_threshold: float = 0.7) -> SearchResultSet:
        """
        Search across all documents.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            categories: Filter by document categories
            similarity_threshold: Minimum similarity score
            
        Returns:
            Search results with relevance scores
            
        Example:
            >>> results = await textrag.search("firewall configuration")
            >>> for result in results.results:
            ...     print(f"{result.document_title}: {result.similarity_score:.3f}")
        """
        return await self.service.search(
            query_text=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            categories=categories
        )
    
    async def search_in_documents(self, 
                                 query: str,
                                 document_ids: List[str],
                                 top_k: int = 5) -> SearchResultSet:
        """
        Search within specific documents.
        
        Args:
            query: Search query text
            document_ids: Documents to search in
            top_k: Number of results to return
            
        Returns:
            Search results from specified documents
        """
        return await self.service.search(
            query_text=query,
            top_k=top_k,
            document_ids=document_ids
        )
    
    async def conversational_search(self, 
                                   query: str,
                                   session_id: str,
                                   context_messages: Optional[List[str]] = None,
                                   top_k: int = 5) -> SearchResultSet:
        """
        Perform context-aware search using conversation history.
        
        Args:
            query: Search query text
            session_id: Conversation session ID
            context_messages: Recent conversation messages
            top_k: Number of results to return
            
        Returns:
            Context-aware search results
            
        Example:
            >>> # In a conversation context
            >>> results = await textrag.conversational_search(
            ...     query="how do I configure it?",
            ...     session_id=session_id,
            ...     context_messages=["Tell me about firewalls"]
            ... )
        """
        return await self.service.context_aware_search(
            query_text=query,
            session_id=session_id,
            conversation_context=context_messages,
            top_k=top_k
        )
    
    # === Advanced Operations ===
    
    async def update_document_processing(self, 
                                        document_id: str,
                                        chunk_size: Optional[int] = None,
                                        chunk_overlap: Optional[int] = None,
                                        chunking_strategy: Optional[str] = None) -> Optional[Document]:
        """
        Update document processing parameters and reprocess.
        
        Args:
            document_id: Document to update
            chunk_size: New chunk size
            chunk_overlap: New chunk overlap  
            chunking_strategy: New chunking strategy
            
        Returns:
            Updated document or None if not found
        """
        strategy = None
        if chunking_strategy:
            strategy = ChunkingStrategy(chunking_strategy)
        
        return await self.service.update_document(
            document_id=document_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunking_strategy=strategy
        )
    
    async def rebuild_embeddings(self, document_id: Optional[str] = None) -> int:
        """
        Rebuild embeddings for documents.
        
        Args:
            document_id: Specific document (or all if None)
            
        Returns:
            Number of documents processed
        """
        return await self.service.rebuild_embeddings(document_id)
    
    # === Analytics and Information ===
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive service statistics.
        
        Returns:
            Dictionary with service statistics
        """
        return self.service.get_service_stats()
    
    def get_document_info(self, document_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document information including processing details
        """
        document = self.service.get_document(document_id)
        if not document:
            return {}
        
        chunks = self.service.repository.get_document_chunks(document_id)
        
        return {
            'document_id': document.document_id,
            'title': document.title,
            'status': document.status.value,
            'character_count': document.character_count,
            'word_count': document.word_count,
            'chunk_count': len(chunks),
            'categories': document.categories,
            'tags': document.tags,
            'quality_score': document.quality_score,
            'created_at': document.created_at.isoformat(),
            'processing_errors': document.processing_errors
        }
    
    # === Maintenance Operations ===
    
    def cleanup_failed_documents(self) -> int:
        """
        Remove documents that failed processing.
        
        Returns:
            Number of documents cleaned up
        """
        return self.service.cleanup_failed_documents()
    
    def clear_all_documents(self) -> bool:
        """
        Remove all documents and data (use with caution).
        
        Returns:
            True if cleared successfully
        """
        try:
            # Clear vector store
            self.service.vector_store.clear()
            
            # Get all documents and delete them
            documents = self.service.list_documents()
            for doc in documents:
                self.service.repository.delete_document(doc.document_id)
            
            self.logger.info("Cleared all TextRAG data")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear all documents: {e}")
            return False
    
    def close(self) -> None:
        """Clean up resources."""
        try:
            self.service.close()
            self.logger.info("TextRAG client closed")
        except Exception as e:
            self.logger.error(f"Error closing TextRAG client: {e}")
    
    # === Integration Helpers ===
    
    async def get_context_for_query(self, 
                                   query: str,
                                   max_context_length: int = 2000) -> str:
        """
        Get relevant context text for a query (useful for RAG pipelines).
        
        Args:
            query: Query to find context for
            max_context_length: Maximum context length in characters
            
        Returns:
            Concatenated relevant text chunks
        """
        results = await self.search(query, top_k=5)
        
        context_parts = []
        total_length = 0
        
        for result in results.results:
            chunk_text = result.content
            if total_length + len(chunk_text) <= max_context_length:
                context_parts.append(chunk_text)
                total_length += len(chunk_text)
            else:
                # Add partial chunk if it fits
                remaining = max_context_length - total_length
                if remaining > 100:  # Only add if meaningful amount remains
                    context_parts.append(chunk_text[:remaining])
                break
        
        return "\n\n".join(context_parts)
    
    # === Factory Methods ===
    
    @classmethod
    def create_with_memory_storage(cls) -> 'TextRAG':
        """Create TextRAG with in-memory storage."""
        return cls(use_persistent_storage=False)
    
    @classmethod
    def create_with_chroma_storage(cls, persist_directory: Optional[str] = None) -> 'TextRAG':
        """Create TextRAG with ChromaDB storage."""
        from .vector_storage import ChromaVectorStore
        
        vector_store = ChromaVectorStore(persist_directory=persist_directory)
        vector_store.initialize()
        
        return cls(vector_store=vector_store)