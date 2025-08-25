"""
REST API endpoints for RAG Orchestrator.

Provides comprehensive API gateway for the hybrid RAG system with
request routing, authentication, rate limiting, and monitoring.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import asyncio
from datetime import datetime
import uuid

from ...shared.models import APIResponse
from ...shared.exceptions import ValidationError, ServiceError
from ...services.rag_orchestrator import RAGOrchestrator, RAGClient
from ...services.rag_orchestrator.models import (
    RAGQuery, RAGResponse, BatchRAGRequest, ProcessingMode, QueryType
)


router = APIRouter(prefix="/api/v1/rag", tags=["RAG Orchestrator"])
security = HTTPBearer(auto_error=False)

# Global orchestrator instance (would be dependency injected in production)
rag_orchestrator: Optional[RAGOrchestrator] = None
rag_client: Optional[RAGClient] = None


def get_rag_orchestrator() -> RAGOrchestrator:
    """Get RAG Orchestrator instance (dependency)."""
    global rag_orchestrator
    if rag_orchestrator is None:
        rag_orchestrator = RAGOrchestrator()
    return rag_orchestrator


def get_rag_client() -> RAGClient:
    """Get RAG Client instance (dependency)."""
    global rag_client
    if rag_client is None:
        rag_client = RAGClient()
    return rag_client


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[Dict[str, Any]]:
    """
    Verify authentication token (placeholder implementation).
    
    In production, this would validate JWT tokens, API keys, etc.
    """
    if not credentials:
        return None
    
    # Placeholder token validation
    # In production, decode JWT, validate API key, etc.
    if credentials.credentials == "demo-token":
        return {"user_id": "demo_user", "permissions": ["read", "write"]}
    
    return None


def check_rate_limit(user_info: Optional[Dict[str, Any]] = None) -> bool:
    """
    Check rate limiting (placeholder implementation).
    
    In production, this would use Redis or similar for rate limiting.
    """
    # Placeholder rate limiting
    # In production, implement sliding window rate limiting
    return True


# Core RAG Operations

@router.post("/query", response_model=APIResponse)
async def query_rag(
    query_data: Dict[str, Any],
    orchestrator: RAGOrchestrator = Depends(get_rag_orchestrator),
    user_info: Optional[Dict[str, Any]] = Depends(verify_token)
):
    """
    Execute a RAG query with full orchestration.
    
    Supports all query types: semantic search, graph traversal, hybrid fusion.
    """
    try:
        # Check rate limiting
        if not check_rate_limit(user_info):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Create RAG query
        rag_query = RAGQuery(
            query_text=query_data.get("query", ""),
            query_type=QueryType(query_data.get("query_type", "hybrid_fusion")),
            processing_mode=ProcessingMode(query_data.get("processing_mode", "balanced")),
            session_id=query_data.get("session_id"),
            user_id=user_info.get("user_id") if user_info else None,
            diagram_id=query_data.get("diagram_id"),
            top_k=query_data.get("top_k", 5),
            similarity_threshold=query_data.get("similarity_threshold", 0.7),
            categories=query_data.get("categories"),
            include_visualizations=query_data.get("include_visualizations", True),
            include_confidence_scores=query_data.get("include_confidence_scores", True)
        )
        
        # Execute query
        response = await orchestrator.query(rag_query)
        
        # Format response
        result_data = {
            "query_id": response.query_id,
            "response_text": response.response_text,
            "confidence_metrics": {
                "overall_confidence": response.confidence_metrics.overall_confidence,
                "reliability_level": response.confidence_metrics.reliability_level.value,
                "source_coverage": response.confidence_metrics.source_coverage,
                "response_grounding": response.confidence_metrics.response_grounding,
                "information_gaps": response.confidence_metrics.information_gaps,
                "confidence_flags": response.confidence_metrics.confidence_flags
            },
            "sources": [
                {
                    "source_id": source.source_id,
                    "title": source.title,
                    "excerpt": source.content_excerpt,
                    "relevance_score": source.relevance_score,
                    "source_type": source.source_type,
                    "metadata": source.metadata
                }
                for source in response.sources
            ],
            "visualizations": [
                {
                    "id": viz.visualization_id,
                    "type": viz.visualization_type,
                    "title": viz.title,
                    "data": {
                        "nodes": viz.nodes,
                        "relationships": viz.relationships
                    },
                    "config": viz.style_config
                }
                for viz in response.visualizations
            ],
            "processing_info": {
                "processing_mode": response.processing_mode.value,
                "processing_time_ms": response.processing_time_ms,
                "requires_verification": response.requires_verification,
                "has_uncertainties": response.has_uncertainties
            },
            "suggestions": {
                "follow_ups": response.suggested_follow_ups,
                "related_queries": response.related_queries
            }
        }
        
        return APIResponse(
            success=True,
            message="Query executed successfully",
            data=result_data
        )
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ServiceError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/ask", response_model=APIResponse)
async def ask_question(
    question_data: Dict[str, Any],
    client: RAGClient = Depends(get_rag_client),
    user_info: Optional[Dict[str, Any]] = Depends(verify_token)
):
    """
    Simple question-answering endpoint with sensible defaults.
    
    Optimized for ease of use with automatic parameter selection.
    """
    try:
        if not check_rate_limit(user_info):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        question = question_data.get("question", "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Execute question
        result = await client.ask(
            question=question,
            session_id=question_data.get("session_id"),
            user_id=user_info.get("user_id") if user_info else None,
            diagram_id=question_data.get("diagram_id"),
            mode=question_data.get("mode", "balanced"),
            top_k=question_data.get("top_k", 5),
            include_visualizations=question_data.get("include_visualizations", True)
        )
        
        return APIResponse(
            success=True,
            message="Question answered successfully",
            data=result
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=APIResponse)
async def search(
    search_data: Dict[str, Any],
    client: RAGClient = Depends(get_rag_client),
    user_info: Optional[Dict[str, Any]] = Depends(verify_token)
):
    """
    Search for relevant information across the knowledge base.
    
    Supports semantic, graph, and hybrid search modes.
    """
    try:
        if not check_rate_limit(user_info):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        query = search_data.get("query", "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="Search query cannot be empty")
        
        # Execute search
        result = await client.search(
            query=query,
            search_type=search_data.get("search_type", "hybrid"),
            top_k=search_data.get("top_k", 10),
            similarity_threshold=search_data.get("similarity_threshold", 0.7),
            categories=search_data.get("categories")
        )
        
        return APIResponse(
            success=True,
            message="Search completed successfully",
            data=result
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Batch Operations

@router.post("/batch/query", response_model=APIResponse)
async def batch_query(
    batch_data: Dict[str, Any],
    orchestrator: RAGOrchestrator = Depends(get_rag_orchestrator),
    user_info: Optional[Dict[str, Any]] = Depends(verify_token)
):
    """
    Process multiple queries in batch for efficient processing.
    """
    try:
        if not check_rate_limit(user_info):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        queries_data = batch_data.get("queries", [])
        if not queries_data:
            raise HTTPException(status_code=400, detail="No queries provided")
        
        if len(queries_data) > 50:  # Reasonable batch size limit
            raise HTTPException(status_code=400, detail="Too many queries in batch (max 50)")
        
        # Create batch request
        queries = []
        for query_data in queries_data:
            rag_query = RAGQuery(
                query_text=query_data.get("query", ""),
                query_type=QueryType(query_data.get("query_type", "hybrid_fusion")),
                processing_mode=ProcessingMode(query_data.get("processing_mode", "balanced")),
                user_id=user_info.get("user_id") if user_info else None,
                top_k=query_data.get("top_k", 5),
                similarity_threshold=query_data.get("similarity_threshold", 0.7)
            )
            queries.append(rag_query)
        
        batch_request = BatchRAGRequest(
            queries=queries,
            max_concurrent=batch_data.get("max_concurrent", 5),
            fail_fast=batch_data.get("fail_fast", False)
        )
        
        # Process batch
        batch_response = await orchestrator.batch_query(batch_request)
        
        # Format response
        result_data = {
            "batch_id": batch_response.batch_id,
            "total_queries": batch_response.total_queries,
            "success_rate": batch_response.success_rate,
            "processing_stats": {
                "avg_processing_time_ms": batch_response.avg_processing_time_ms,
                "total_processing_time_ms": batch_response.total_processing_time_ms,
                "avg_confidence": batch_response.avg_confidence
            },
            "successful_responses": len(batch_response.successful_responses),
            "failed_queries": len(batch_response.failed_queries),
            "results": [
                {
                    "query_id": resp.query_id,
                    "response_text": resp.response_text,
                    "confidence": resp.confidence_metrics.overall_confidence,
                    "processing_time_ms": resp.processing_time_ms
                }
                for resp in batch_response.successful_responses
            ]
        }
        
        return APIResponse(
            success=True,
            message=f"Batch processing completed: {batch_response.success_rate:.1%} success rate",
            data=result_data
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch/ask", response_model=APIResponse)
async def batch_ask(
    batch_data: Dict[str, Any],
    client: RAGClient = Depends(get_rag_client),
    user_info: Optional[Dict[str, Any]] = Depends(verify_token)
):
    """
    Ask multiple questions in batch with simplified interface.
    """
    try:
        if not check_rate_limit(user_info):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        questions = batch_data.get("questions", [])
        if not questions:
            raise HTTPException(status_code=400, detail="No questions provided")
        
        if len(questions) > 20:  # Reasonable limit for simplified interface
            raise HTTPException(status_code=400, detail="Too many questions in batch (max 20)")
        
        # Process batch
        results = await client.batch_ask(
            questions=questions,
            session_id=batch_data.get("session_id"),
            parallel=batch_data.get("parallel", True),
            max_concurrent=batch_data.get("max_concurrent", 3)
        )
        
        return APIResponse(
            success=True,
            message="Batch questions processed successfully",
            data={"results": results}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Asynchronous Operations

@router.post("/async/query", response_model=APIResponse)
async def start_async_query(
    query_data: Dict[str, Any],
    orchestrator: RAGOrchestrator = Depends(get_rag_orchestrator),
    user_info: Optional[Dict[str, Any]] = Depends(verify_token)
):
    """
    Start asynchronous query processing for long-running operations.
    """
    try:
        if not check_rate_limit(user_info):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Create RAG query
        rag_query = RAGQuery(
            query_text=query_data.get("query", ""),
            query_type=QueryType(query_data.get("query_type", "hybrid_fusion")),
            processing_mode=ProcessingMode(query_data.get("processing_mode", "comprehensive")),
            user_id=user_info.get("user_id") if user_info else None,
            top_k=query_data.get("top_k", 10),
            timeout_seconds=query_data.get("timeout_seconds", 60.0)
        )
        
        # Start async processing
        operation_id = await orchestrator.query_async(rag_query)
        
        return APIResponse(
            success=True,
            message="Async query started successfully",
            data={
                "operation_id": operation_id,
                "status_url": f"/api/v1/rag/async/status/{operation_id}",
                "estimated_completion": "30-60 seconds"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/async/status/{operation_id}", response_model=APIResponse)
async def get_async_status(
    operation_id: str,
    orchestrator: RAGOrchestrator = Depends(get_rag_orchestrator),
    user_info: Optional[Dict[str, Any]] = Depends(verify_token)
):
    """
    Get status of asynchronous operation.
    """
    try:
        status = await orchestrator.get_query_status(operation_id)
        
        result_data = {
            "operation_id": status.operation_id,
            "status": status.status,
            "progress": status.progress,
            "current_stage": status.current_stage,
            "completed_stages": status.completed_stages,
            "processing_time_ms": status.processing_time_ms,
            "estimated_completion": status.estimated_completion.isoformat() if status.estimated_completion else None,
            "error": status.error
        }
        
        # Include result if completed
        if status.status == "completed" and status.result:
            result_data["result"] = {
                "response_text": status.result.response_text,
                "confidence": status.result.confidence_metrics.overall_confidence,
                "sources_count": len(status.result.sources)
            }
        
        return APIResponse(
            success=True,
            message=f"Operation status: {status.status}",
            data=result_data
        )
        
    except ValidationError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Knowledge Base Management

@router.post("/documents/add", response_model=APIResponse)
async def add_document(
    content: str = Form(...),
    title: str = Form(...),
    document_type: str = Form("text"),
    categories: Optional[str] = Form(None),  # JSON string
    metadata: Optional[str] = Form(None),  # JSON string
    orchestrator: RAGOrchestrator = Depends(get_rag_orchestrator),
    user_info: Optional[Dict[str, Any]] = Depends(verify_token)
):
    """
    Add a text document to the knowledge base.
    """
    try:
        if not user_info or "write" not in user_info.get("permissions", []):
            raise HTTPException(status_code=403, detail="Write permission required")
        
        if not check_rate_limit(user_info):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Parse optional JSON fields
        import json
        doc_categories = json.loads(categories) if categories else []
        doc_metadata = json.loads(metadata) if metadata else {}
        
        # Add document
        result = await orchestrator.add_document(
            content=content,
            title=title,
            document_type=document_type,
            metadata={**doc_metadata, "categories": doc_categories}
        )
        
        return APIResponse(
            success=True,
            message="Document added successfully",
            data=result
        )
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in categories or metadata")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/upload", response_model=APIResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    document_type: str = Form("document"),
    client: RAGClient = Depends(get_rag_client),
    user_info: Optional[Dict[str, Any]] = Depends(verify_token)
):
    """
    Upload and process a document file.
    """
    try:
        if not user_info or "write" not in user_info.get("permissions", []):
            raise HTTPException(status_code=403, detail="Write permission required")
        
        if not check_rate_limit(user_info):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Read file content
        content = await file.read()
        content_str = content.decode('utf-8')
        
        # Use filename as title if not provided
        doc_title = title or file.filename or "Uploaded Document"
        
        # Add document
        result = await client.add_document(
            content=content_str,
            title=doc_title,
            document_type=document_type,
            metadata={"filename": file.filename, "content_type": file.content_type}
        )
        
        return APIResponse(
            success=True,
            message="Document uploaded and processed successfully",
            data=result
        )
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be text-based")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/diagrams/process", response_model=APIResponse)
async def process_diagram(
    image_path: str = Form(...),
    diagram_id: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    orchestrator: RAGOrchestrator = Depends(get_rag_orchestrator),
    user_info: Optional[Dict[str, Any]] = Depends(verify_token)
):
    """
    Process a diagram and add it to the knowledge base.
    """
    try:
        if not user_info or "write" not in user_info.get("permissions", []):
            raise HTTPException(status_code=403, detail="Write permission required")
        
        if not check_rate_limit(user_info):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Generate diagram ID if not provided
        if not diagram_id:
            diagram_id = f"diagram_{uuid.uuid4().hex[:12]}"
        
        # Process diagram
        result = await orchestrator.process_diagram(
            image_path=image_path,
            diagram_id=diagram_id,
            title=title
        )
        
        return APIResponse(
            success=True,
            message="Diagram processed successfully",
            data=result
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Conversation Management

@router.post("/conversations/start", response_model=APIResponse)
async def start_conversation(
    conversation_data: Dict[str, Any],
    client: RAGClient = Depends(get_rag_client),
    user_info: Optional[Dict[str, Any]] = Depends(verify_token)
):
    """
    Start a new conversation session.
    """
    try:
        if not check_rate_limit(user_info):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        session_id = await client.start_conversation(
            user_id=user_info.get("user_id") if user_info else None,
            preferences=conversation_data.get("preferences", {})
        )
        
        return APIResponse(
            success=True,
            message="Conversation started successfully",
            data={"session_id": session_id}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversations/{session_id}/message", response_model=APIResponse)
async def send_message(
    session_id: str,
    message_data: Dict[str, Any],
    client: RAGClient = Depends(get_rag_client),
    user_info: Optional[Dict[str, Any]] = Depends(verify_token)
):
    """
    Send a message in an existing conversation.
    """
    try:
        if not check_rate_limit(user_info):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        message = message_data.get("message", "").strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        result = await client.continue_conversation(
            session_id=session_id,
            message=message,
            mode=message_data.get("mode", "interactive")
        )
        
        return APIResponse(
            success=True,
            message="Message processed successfully",
            data=result
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# System Management

@router.get("/system/status", response_model=APIResponse)
async def get_system_status(
    client: RAGClient = Depends(get_rag_client),
    user_info: Optional[Dict[str, Any]] = Depends(verify_token)
):
    """
    Get comprehensive system health and status information.
    """
    try:
        status = client.get_system_health()
        
        return APIResponse(
            success=True,
            message="System status retrieved successfully",
            data=status
        )
        
    except Exception as e:
        return APIResponse(
            success=False,
            message="Failed to get system status",
            data={"error": str(e)}
        )


@router.get("/system/metrics", response_model=APIResponse)
async def get_system_metrics(
    orchestrator: RAGOrchestrator = Depends(get_rag_orchestrator),
    user_info: Optional[Dict[str, Any]] = Depends(verify_token)
):
    """
    Get detailed system performance metrics.
    """
    try:
        if not user_info or user_info.get("user_id") != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        
        metrics = {
            "performance_stats": orchestrator.performance_stats,
            "reliability_report": orchestrator.reliability_manager.get_system_reliability_report(),
            "active_operations": len(orchestrator.active_operations),
            "timestamp": datetime.now().isoformat()
        }
        
        return APIResponse(
            success=True,
            message="System metrics retrieved successfully",
            data=metrics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Health Check

@router.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}