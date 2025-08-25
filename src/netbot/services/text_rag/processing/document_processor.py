"""
Main document processing pipeline for TextRAG.

Orchestrates the complete document processing workflow from ingestion
to chunking and embedding preparation.
"""

import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

from ....shared import get_logger
from ..models import Document, DocumentChunk, DocumentType, DocumentStatus
from ..repository import TextRAGRepository
from .content_extractor import ContentExtractor
from .text_chunker import TextChunker, ChunkConfig


class DocumentProcessor:
    """
    Main document processing pipeline.
    
    Handles the complete workflow from document ingestion through
    chunking and metadata extraction, ready for vector storage.
    """
    
    def __init__(self, repository: Optional[TextRAGRepository] = None):
        self.logger = get_logger(__name__)
        self.repository = repository or TextRAGRepository()
        self.content_extractor = ContentExtractor()
        self.text_chunker = TextChunker()
        
        self.logger.info("Document Processor initialized")
    
    async def process_file(self, 
                          file_path: Union[str, Path],
                          title: Optional[str] = None,
                          categories: Optional[List[str]] = None,
                          tags: Optional[List[str]] = None,
                          chunk_config: Optional[ChunkConfig] = None) -> Document:
        """
        Process a file from disk.
        
        Args:
            file_path: Path to the file to process
            title: Document title (defaults to filename)
            categories: Document categories
            tags: Document tags
            chunk_config: Chunking configuration
            
        Returns:
            Processed document with chunks
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Extract content from file
        content, extraction_metadata = self.content_extractor.extract_from_file(str(file_path))
        
        # Determine document type
        doc_type = self._get_document_type_from_path(file_path)
        
        # Create document
        document = Document(
            title=title or file_path.stem,
            content=content,
            document_type=doc_type,
            source_path=str(file_path),
            categories=categories or [],
            tags=tags or [],
            status=DocumentStatus.PROCESSING
        )
        
        # Add extraction metadata
        document.metadata.update(extraction_metadata)
        document.file_size = file_path.stat().st_size
        
        return await self._process_document(document, chunk_config)
    
    async def process_content(self, 
                             content: str,
                             title: str,
                             document_type: DocumentType,
                             source_url: Optional[str] = None,
                             author: Optional[str] = None,
                             categories: Optional[List[str]] = None,
                             tags: Optional[List[str]] = None,
                             chunk_config: Optional[ChunkConfig] = None) -> Document:
        """
        Process content directly (e.g., from web scraping or API).
        
        Args:
            content: Raw text content
            title: Document title
            document_type: Type of document
            source_url: Source URL if applicable
            author: Document author
            categories: Document categories
            tags: Document tags
            chunk_config: Chunking configuration
            
        Returns:
            Processed document with chunks
        """
        # Extract and clean content
        processed_content, extraction_metadata = self.content_extractor.extract_from_content(
            content, document_type
        )
        
        # Create document
        document = Document(
            title=title,
            content=processed_content,
            document_type=document_type,
            source_url=source_url,
            author=author,
            categories=categories or [],
            tags=tags or [],
            status=DocumentStatus.PROCESSING
        )
        
        # Add extraction metadata
        document.metadata.update(extraction_metadata)
        
        return await self._process_document(document, chunk_config)
    
    async def reprocess_document(self, 
                                document_id: str,
                                chunk_config: Optional[ChunkConfig] = None) -> Optional[Document]:
        """
        Reprocess an existing document with new chunking configuration.
        
        Args:
            document_id: Document to reprocess
            chunk_config: New chunking configuration
            
        Returns:
            Reprocessed document or None if not found
        """
        document = self.repository.get_document(document_id)
        if not document:
            self.logger.error(f"Document not found: {document_id}")
            return None
        
        # Delete existing chunks
        self.repository.delete_document_chunks(document_id)
        
        # Reset status
        document.status = DocumentStatus.PROCESSING
        document.processing_errors = []
        
        return await self._process_document(document, chunk_config, update_existing=True)
    
    async def _process_document(self, 
                               document: Document,
                               chunk_config: Optional[ChunkConfig] = None,
                               update_existing: bool = False) -> Document:
        """
        Internal method to process a document.
        
        Args:
            document: Document to process
            chunk_config: Chunking configuration
            update_existing: Whether this is updating an existing document
            
        Returns:
            Processed document
        """
        try:
            # Use default chunk config if not provided
            if chunk_config is None:
                chunk_config = ChunkConfig(
                    strategy=document.chunking_strategy,
                    chunk_size=document.chunk_size,
                    chunk_overlap=document.chunk_overlap
                )
            
            # Update document with chunk config
            document.chunking_strategy = chunk_config.strategy
            document.chunk_size = chunk_config.chunk_size
            document.chunk_overlap = chunk_config.chunk_overlap
            
            # Save or update document
            if update_existing:
                self.repository.update_document(document)
            else:
                self.repository.create_document(document)
            
            # Create chunks
            chunks = self.text_chunker.chunk_document(
                document.document_id,
                document.content,
                chunk_config
            )
            
            # Save chunks to repository
            for chunk in chunks:
                self.repository.create_chunk(chunk)
            
            # Calculate quality score
            quality_score = self._calculate_document_quality(document, chunks)
            document.quality_score = quality_score
            
            # Update status to completed
            document.status = DocumentStatus.COMPLETED
            self.repository.update_document(document)
            
            self.logger.info(f"Successfully processed document {document.document_id} with {len(chunks)} chunks")
            return document
            
        except Exception as e:
            self.logger.error(f"Failed to process document {document.document_id}: {e}")
            
            # Mark as failed and record error
            document.status = DocumentStatus.FAILED
            document.mark_processing_error(str(e))
            
            if update_existing:
                self.repository.update_document(document)
            else:
                try:
                    self.repository.create_document(document)
                except:
                    pass  # Document might already exist
            
            raise
    
    async def batch_process(self, 
                           file_paths: List[Union[str, Path]],
                           categories: Optional[List[str]] = None,
                           tags: Optional[List[str]] = None,
                           chunk_config: Optional[ChunkConfig] = None,
                           max_concurrent: int = 5) -> List[Document]:
        """
        Process multiple files concurrently.
        
        Args:
            file_paths: List of file paths to process
            categories: Default categories for all documents
            tags: Default tags for all documents  
            chunk_config: Chunking configuration
            max_concurrent: Maximum concurrent processing tasks
            
        Returns:
            List of processed documents
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_file(file_path):
            async with semaphore:
                try:
                    return await self.process_file(
                        file_path=file_path,
                        categories=categories,
                        tags=tags,
                        chunk_config=chunk_config
                    )
                except Exception as e:
                    self.logger.error(f"Failed to process file {file_path}: {e}")
                    return None
        
        tasks = [process_single_file(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None results and exceptions
        documents = [doc for doc in results if isinstance(doc, Document)]
        
        self.logger.info(f"Batch processed {len(documents)} out of {len(file_paths)} files")
        return documents
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        return self.repository.get_document_statistics()
    
    def _get_document_type_from_path(self, file_path: Path) -> DocumentType:
        """Determine document type from file extension."""
        extension = file_path.suffix.lower()
        
        type_mapping = {
            '.txt': DocumentType.TEXT,
            '.md': DocumentType.MARKDOWN,
            '.markdown': DocumentType.MARKDOWN,
            '.html': DocumentType.HTML,
            '.htm': DocumentType.HTML,
            '.pdf': DocumentType.PDF,
            '.doc': DocumentType.WORD,
            '.docx': DocumentType.WORD,
            '.ppt': DocumentType.POWERPOINT,
            '.pptx': DocumentType.POWERPOINT,
            '.csv': DocumentType.CSV,
            '.json': DocumentType.JSON,
        }
        
        return type_mapping.get(extension, DocumentType.TEXT)
    
    def _calculate_document_quality(self, document: Document, chunks: List[DocumentChunk]) -> float:
        """
        Calculate a quality score for the document based on various metrics.
        
        Args:
            document: The document
            chunks: List of created chunks
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not chunks:
            return 0.0
        
        quality_factors = []
        
        # 1. Content length factor (optimal around 1000-5000 chars)
        content_length = len(document.content)
        if content_length < 100:
            length_score = content_length / 100.0
        elif content_length <= 5000:
            length_score = 1.0
        else:
            length_score = max(0.5, 5000.0 / content_length)
        quality_factors.append(length_score)
        
        # 2. Chunk consistency factor
        chunk_sizes = [len(chunk.content) for chunk in chunks]
        if chunk_sizes:
            avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)
            size_variance = sum((size - avg_chunk_size) ** 2 for size in chunk_sizes) / len(chunk_sizes)
            consistency_score = 1.0 / (1.0 + size_variance / 10000.0)  # Normalize variance
            quality_factors.append(consistency_score)
        
        # 3. Coherence factor (average of chunk coherence scores)
        coherence_scores = [chunk.coherence_score for chunk in chunks if chunk.coherence_score > 0]
        if coherence_scores:
            avg_coherence = sum(coherence_scores) / len(coherence_scores)
            quality_factors.append(avg_coherence)
        
        # 4. Metadata richness factor
        metadata_score = 0.0
        if document.author:
            metadata_score += 0.2
        if document.tags:
            metadata_score += 0.3
        if document.categories:
            metadata_score += 0.3
        if document.source_path or document.source_url:
            metadata_score += 0.2
        quality_factors.append(metadata_score)
        
        # Calculate overall quality score
        overall_quality = sum(quality_factors) / len(quality_factors) if quality_factors else 0.0
        return min(1.0, max(0.0, overall_quality))