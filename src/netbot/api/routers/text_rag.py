"""
TextRAG API endpoints for NetBot V2.

Provides REST API endpoints for document management, semantic search,
and text-based retrieval operations.
"""

import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, Query, Body
from pydantic import BaseModel, Field

from ...shared import get_logger
from ...services.text_rag import TextRAGService
from ..models import APIResponse, ErrorResponse

# Initialize router and service
router = APIRouter()
logger = get_logger(__name__)

# Initialize TextRAG service
try:
    text_rag_service = TextRAGService()
    logger.info("TextRAG API endpoints initialized")
except Exception as e:
    logger.error(f"Failed to initialize TextRAG service: {e}")
    text_rag_service = None


# === API Models ===

class DocumentUploadRequest(BaseModel):
    """Request model for document upload."""
    title: Optional[str] = Field(None, description="Document title")
    categories: Optional[List[str]] = Field(default_factory=list, description="Document categories")
    tags: Optional[List[str]] = Field(default_factory=list, description="Document tags")
    chunk_size: int = Field(1000, ge=100, le=5000, description="Chunk size for processing")
    chunk_overlap: int = Field(200, ge=0, le=1000, description="Chunk overlap")
    chunking_strategy: str = Field("recursive_character", description="Chunking strategy")


class TextDocumentRequest(BaseModel):
    """Request model for text document creation."""
    content: str = Field(..., description="Document text content")
    title: str = Field(..., description="Document title")
    document_type: str = Field("text", description="Document type")
    source_url: Optional[str] = Field(None, description="Source URL")
    author: Optional[str] = Field(None, description="Document author")
    categories: Optional[List[str]] = Field(default_factory=list, description="Document categories")
    tags: Optional[List[str]] = Field(default_factory=list, description="Document tags")
    chunk_size: int = Field(1000, ge=100, le=5000, description="Chunk size for processing")
    chunk_overlap: int = Field(200, ge=0, le=1000, description="Chunk overlap")
    chunking_strategy: str = Field("recursive_character", description="Chunking strategy")


class SearchRequest(BaseModel):
    """Request model for text search."""
    query: str = Field(..., description="Search query text")
    top_k: int = Field(5, ge=1, le=50, description="Number of results to return")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    document_ids: Optional[List[str]] = Field(None, description="Limit search to specific documents")
    categories: Optional[List[str]] = Field(None, description="Filter by categories")
    include_metadata: bool = Field(True, description="Include result metadata")


class ConversationalSearchRequest(BaseModel):
    """Request model for conversational search."""
    query: str = Field(..., description="Search query text")
    session_id: str = Field(..., description="Conversation session ID")
    context_messages: Optional[List[str]] = Field(None, description="Recent conversation messages")
    top_k: int = Field(5, ge=1, le=50, description="Number of results to return")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score")


class DocumentUpdateRequest(BaseModel):
    """Request model for document updates."""
    chunk_size: Optional[int] = Field(None, ge=100, le=5000, description="New chunk size")
    chunk_overlap: Optional[int] = Field(None, ge=0, le=1000, description="New chunk overlap")
    chunking_strategy: Optional[str] = Field(None, description="New chunking strategy")


class BatchUploadRequest(BaseModel):
    """Request model for batch document processing."""
    categories: Optional[List[str]] = Field(default_factory=list, description="Default categories")
    tags: Optional[List[str]] = Field(default_factory=list, description="Default tags")
    chunk_size: int = Field(1000, ge=100, le=5000, description="Chunk size for processing")
    chunk_overlap: int = Field(200, ge=0, le=1000, description="Chunk overlap")


# === Document Management Endpoints ===

@router.post("/documents/upload", response_model=APIResponse)
async def upload_document(
    file: UploadFile = File(...),
    request: DocumentUploadRequest = Body(default_factory=DocumentUploadRequest)
):
    """
    Upload and process a document file.
    
    Supports various file formats including PDF, Word, Text, Markdown, etc.
    """
    if not text_rag_service:
        raise HTTPException(status_code=503, detail="TextRAG service not available")
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Process the document
            from ...services.text_rag.models import ChunkingStrategy
            
            document = await text_rag_service.add_document_from_file(
                file_path=temp_file_path,
                title=request.title or Path(file.filename).stem,
                categories=request.categories,
                tags=request.tags,
                chunking_strategy=ChunkingStrategy(request.chunking_strategy),
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap
            )
            
            return APIResponse(
                success=True,
                data={
                    "document_id": document.document_id,
                    "title": document.title,
                    "status": document.status.value,
                    "character_count": document.character_count,
                    "word_count": document.word_count,
                    "categories": document.categories,
                    "tags": document.tags
                },
                message="Document uploaded and processed successfully"
            )
            
        finally:
            # Clean up temporary file
            Path(temp_file_path).unlink(missing_ok=True)
    
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")


@router.post("/documents/text", response_model=APIResponse)
async def create_text_document(request: TextDocumentRequest):
    """Create a document from text content."""
    if not text_rag_service:
        raise HTTPException(status_code=503, detail="TextRAG service not available")
    
    try:
        from ...services.text_rag.models import ChunkingStrategy
        
        document = await text_rag_service.add_document_from_content(
            content=request.content,
            title=request.title,
            document_type=request.document_type,
            source_url=request.source_url,
            author=request.author,
            categories=request.categories,
            tags=request.tags,
            chunking_strategy=ChunkingStrategy(request.chunking_strategy),
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        
        return APIResponse(
            success=True,
            data={
                "document_id": document.document_id,
                "title": document.title,
                "status": document.status.value,
                "character_count": document.character_count,
                "word_count": document.word_count,
                "categories": document.categories,
                "tags": document.tags
            },
            message="Text document created successfully"
        )
    
    except Exception as e:
        logger.error(f"Text document creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document creation failed: {str(e)}")


@router.get("/documents/{document_id}", response_model=APIResponse)
async def get_document(document_id: str):
    """Get a document by ID."""
    if not text_rag_service:
        raise HTTPException(status_code=503, detail="TextRAG service not available")
    
    try:
        document = text_rag_service.get_document(document_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get chunks for additional info
        chunks = text_rag_service.repository.get_document_chunks(document_id)
        
        return APIResponse(
            success=True,
            data={
                "document_id": document.document_id,
                "title": document.title,
                "document_type": document.document_type.value,
                "status": document.status.value,
                "character_count": document.character_count,
                "word_count": document.word_count,
                "chunk_count": len(chunks),
                "categories": document.categories,
                "tags": document.tags,
                "source_path": document.source_path,
                "source_url": document.source_url,
                "author": document.author,
                "quality_score": document.quality_score,
                "processing_errors": document.processing_errors,
                "created_at": document.created_at.isoformat(),
                "updated_at": document.updated_at.isoformat() if document.updated_at else None
            },
            message="Document retrieved successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve document: {str(e)}")


@router.get("/documents", response_model=APIResponse)
async def list_documents(
    categories: Optional[List[str]] = Query(None, description="Filter by categories"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of documents")
):
    """List documents with optional filtering."""
    if not text_rag_service:
        raise HTTPException(status_code=503, detail="TextRAG service not available")
    
    try:
        from ...services.text_rag.models import DocumentStatus
        
        status_filter = None
        if status:
            try:
                status_filter = DocumentStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        documents = text_rag_service.list_documents(
            status=status_filter,
            categories=categories,
            limit=limit
        )
        
        document_list = []
        for doc in documents:
            document_list.append({
                "document_id": doc.document_id,
                "title": doc.title,
                "document_type": doc.document_type.value,
                "status": doc.status.value,
                "character_count": doc.character_count,
                "word_count": doc.word_count,
                "categories": doc.categories,
                "tags": doc.tags,
                "quality_score": doc.quality_score,
                "created_at": doc.created_at.isoformat()
            })
        
        return APIResponse(
            success=True,
            data={
                "documents": document_list,
                "count": len(document_list),
                "total_available": len(document_list)  # This could be optimized
            },
            message=f"Retrieved {len(document_list)} documents"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.put("/documents/{document_id}", response_model=APIResponse)
async def update_document(document_id: str, request: DocumentUpdateRequest):
    """Update document processing parameters and reprocess."""
    if not text_rag_service:
        raise HTTPException(status_code=503, detail="TextRAG service not available")
    
    try:
        from ...services.text_rag.models import ChunkingStrategy
        
        chunking_strategy = None
        if request.chunking_strategy:
            chunking_strategy = ChunkingStrategy(request.chunking_strategy)
        
        document = await text_rag_service.update_document(
            document_id=document_id,
            chunking_strategy=chunking_strategy,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return APIResponse(
            success=True,
            data={
                "document_id": document.document_id,
                "title": document.title,
                "status": document.status.value,
                "character_count": document.character_count,
                "word_count": document.word_count
            },
            message="Document updated and reprocessed successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Document update failed: {str(e)}")


@router.delete("/documents/{document_id}", response_model=APIResponse)
async def delete_document(document_id: str):
    """Delete a document and all its chunks."""
    if not text_rag_service:
        raise HTTPException(status_code=503, detail="TextRAG service not available")
    
    try:
        success = await text_rag_service.delete_document(document_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return APIResponse(
            success=True,
            data={"document_id": document_id},
            message="Document deleted successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Document deletion failed: {str(e)}")


# === Search Endpoints ===

@router.post("/search", response_model=APIResponse)
async def search_documents(request: SearchRequest):
    """Search across all documents using semantic similarity."""
    if not text_rag_service:
        raise HTTPException(status_code=503, detail="TextRAG service not available")
    
    try:
        result_set = await text_rag_service.search(
            query_text=request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            document_ids=request.document_ids,
            categories=request.categories
        )
        
        # Format results
        search_results = []
        for result in result_set.results:
            result_data = result.to_dict()
            if not request.include_metadata:
                # Remove metadata fields to reduce response size
                result_data.pop('metadata', None)
            
            search_results.append(result_data)
        
        return APIResponse(
            success=True,
            data={
                "query": result_set.query_text,
                "results": search_results,
                "total_results": result_set.total_results,
                "returned_results": result_set.returned_results,
                "search_time_ms": result_set.search_time_ms,
                "search_method": result_set.search_method.value,
                "filters_applied": result_set.filters_applied
            },
            message=f"Found {result_set.returned_results} results"
        )
    
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/search/conversational", response_model=APIResponse)
async def conversational_search(request: ConversationalSearchRequest):
    """Perform context-aware search using conversation history."""
    if not text_rag_service:
        raise HTTPException(status_code=503, detail="TextRAG service not available")
    
    try:
        result_set = await text_rag_service.context_aware_search(
            query_text=request.query,
            session_id=request.session_id,
            conversation_context=request.context_messages,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )
        
        # Format results
        search_results = [result.to_dict() for result in result_set.results]
        
        return APIResponse(
            success=True,
            data={
                "query": result_set.query_text,
                "session_id": request.session_id,
                "results": search_results,
                "total_results": result_set.total_results,
                "returned_results": result_set.returned_results,
                "search_time_ms": result_set.search_time_ms,
                "search_method": result_set.search_method.value
            },
            message=f"Context-aware search found {result_set.returned_results} results"
        )
    
    except Exception as e:
        logger.error(f"Conversational search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Conversational search failed: {str(e)}")


@router.get("/search/context", response_model=APIResponse)
async def get_context_for_query(
    query: str = Query(..., description="Query to find context for"),
    max_length: int = Query(2000, ge=100, le=10000, description="Maximum context length")
):
    """Get relevant context text for a query (useful for RAG pipelines)."""
    if not text_rag_service:
        raise HTTPException(status_code=503, detail="TextRAG service not available")
    
    try:
        results = await text_rag_service.search(query_text=query, top_k=5)
        
        context_parts = []
        total_length = 0
        
        for result in results.results:
            chunk_text = result.content
            if total_length + len(chunk_text) <= max_length:
                context_parts.append(chunk_text)
                total_length += len(chunk_text)
            else:
                # Add partial chunk if it fits
                remaining = max_length - total_length
                if remaining > 100:  # Only add if meaningful amount remains
                    context_parts.append(chunk_text[:remaining])
                break
        
        context_text = "\n\n".join(context_parts)
        
        return APIResponse(
            success=True,
            data={
                "query": query,
                "context": context_text,
                "context_length": len(context_text),
                "chunks_used": len(context_parts),
                "total_results_found": results.total_results
            },
            message="Context retrieved successfully"
        )
    
    except Exception as e:
        logger.error(f"Context retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Context retrieval failed: {str(e)}")


# === Batch Operations ===

@router.post("/documents/batch", response_model=APIResponse)
async def batch_upload_documents(
    files: List[UploadFile] = File(...),
    request: BatchUploadRequest = Body(default_factory=BatchUploadRequest)
):
    """Upload and process multiple documents at once."""
    if not text_rag_service:
        raise HTTPException(status_code=503, detail="TextRAG service not available")
    
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        temp_files = []
        try:
            # Save all uploaded files temporarily
            for file in files:
                if not file.filename:
                    continue
                
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, 
                    suffix=Path(file.filename).suffix
                )
                content = await file.read()
                temp_file.write(content)
                temp_file.close()
                temp_files.append(temp_file.name)
            
            if not temp_files:
                raise HTTPException(status_code=400, detail="No valid files provided")
            
            # Process documents
            documents = await text_rag_service.batch_add_documents(
                file_paths=temp_files,
                categories=request.categories,
                tags=request.tags
            )
            
            # Format response
            processed_documents = []
            for doc in documents:
                processed_documents.append({
                    "document_id": doc.document_id,
                    "title": doc.title,
                    "status": doc.status.value,
                    "character_count": doc.character_count,
                    "word_count": doc.word_count
                })
            
            return APIResponse(
                success=True,
                data={
                    "processed_documents": processed_documents,
                    "successful_count": len(processed_documents),
                    "total_uploaded": len(files)
                },
                message=f"Processed {len(processed_documents)} out of {len(files)} files"
            )
        
        finally:
            # Clean up temporary files
            for temp_file_path in temp_files:
                Path(temp_file_path).unlink(missing_ok=True)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch upload failed: {str(e)}")


# === Analytics and Maintenance ===

@router.get("/stats", response_model=APIResponse)
async def get_service_stats():
    """Get comprehensive TextRAG service statistics."""
    if not text_rag_service:
        raise HTTPException(status_code=503, detail="TextRAG service not available")
    
    try:
        stats = text_rag_service.get_service_stats()
        
        return APIResponse(
            success=True,
            data=stats,
            message="Service statistics retrieved successfully"
        )
    
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve statistics: {str(e)}")


@router.post("/maintenance/rebuild-embeddings", response_model=APIResponse)
async def rebuild_embeddings(document_id: Optional[str] = Query(None, description="Specific document ID")):
    """Rebuild embeddings for documents."""
    if not text_rag_service:
        raise HTTPException(status_code=503, detail="TextRAG service not available")
    
    try:
        processed_count = await text_rag_service.rebuild_embeddings(document_id)
        
        return APIResponse(
            success=True,
            data={
                "documents_processed": processed_count,
                "document_id": document_id
            },
            message=f"Rebuilt embeddings for {processed_count} document(s)"
        )
    
    except Exception as e:
        logger.error(f"Failed to rebuild embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to rebuild embeddings: {str(e)}")


@router.delete("/maintenance/cleanup-failed", response_model=APIResponse)
async def cleanup_failed_documents():
    """Remove documents that failed processing."""
    if not text_rag_service:
        raise HTTPException(status_code=503, detail="TextRAG service not available")
    
    try:
        cleaned_count = text_rag_service.cleanup_failed_documents()
        
        return APIResponse(
            success=True,
            data={"cleaned_documents": cleaned_count},
            message=f"Cleaned up {cleaned_count} failed documents"
        )
    
    except Exception as e:
        logger.error(f"Failed to cleanup failed documents: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")