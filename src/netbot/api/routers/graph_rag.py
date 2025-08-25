"""
Graph RAG API endpoints.
"""

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ...services.graph_rag import GraphRAGService
from ...services.graph_rag.models import SearchRequest, VisualizationRequest
from ..models import APIResponse, ErrorResponse

router = APIRouter()

# Initialize service
graph_rag_service = GraphRAGService()


class SearchRequestAPI(BaseModel):
    """API request model for graph search."""
    query: str
    diagram_id: str
    method: str = "auto"
    top_k: int = 8
    min_similarity: float = 0.1
    include_explanation: bool = False
    detailed_explanation: bool = False
    include_visualization: bool = False


class VisualizationRequestAPI(BaseModel):
    """API request model for visualization."""
    backend: str = "graphviz"
    layout: Optional[str] = None
    format: str = "svg"
    show_node_properties: bool = True
    show_edge_properties: bool = True
    width: int = 1200
    height: int = 800
    node_size: int = 1000
    font_size: int = 12


@router.post("/search")
async def search_graph(request: SearchRequestAPI):
    """
    Search the knowledge graph with natural language.
    
    Args:
        request: Search parameters
        
    Returns:
        Search results with nodes, relationships, and optional explanation/visualization
    """
    try:
        # Convert to service request
        search_request = SearchRequest(
            query=request.query,
            diagram_id=request.diagram_id,
            method=request.method,
            top_k=request.top_k,
            min_similarity=request.min_similarity,
            include_explanation=request.include_explanation,
            detailed_explanation=request.detailed_explanation,
            include_visualization=request.include_visualization
        )
        
        # Perform search
        result = graph_rag_service.search(search_request)
        
        # Prepare response data
        response_data = {
            "query": result.request.query,
            "diagram_id": result.request.diagram_id,
            "method": result.request.method,
            "success": result.success,
            "nodes": [node.dict() for node in result.nodes],
            "relationships": [rel.dict() for rel in result.relationships],
            "node_count": len(result.nodes),
            "relationship_count": len(result.relationships),
            "relevance_score": result.relevance_score,
            "confidence_score": result.confidence_score,
            "search_time_ms": result.metadata.search_time_ms,
            "explanation": result.explanation,
            "visualization_data": result.visualization_data,
            "error_message": result.error_message
        }
        
        if not result.success:
            return JSONResponse(
                status_code=422,
                content=ErrorResponse(
                    error="SearchError",
                    message=result.error_message or "Search failed",
                    timestamp=datetime.utcnow().isoformat() + "Z"
                ).dict()
            )
        
        return APIResponse(
            success=True,
            data=response_data,
            message="Graph search completed successfully",
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=type(e).__name__,
                message=str(e),
                timestamp=datetime.utcnow().isoformat() + "Z"
            ).dict()
        )


@router.post("/search/{diagram_id}")
async def search_diagram(
    diagram_id: str,
    query: str = Query(..., description="Natural language search query"),
    method: str = Query("auto", description="Search method: auto, vector, cypher, hybrid"),
    top_k: int = Query(8, description="Number of results to return"),
    min_similarity: float = Query(0.1, description="Minimum similarity threshold"),
    include_explanation: bool = Query(False, description="Include explanation"),
    detailed_explanation: bool = Query(False, description="Use detailed explanation"),
    include_visualization: bool = Query(False, description="Include visualization")
):
    """
    Search a specific diagram with natural language query.
    
    Args:
        diagram_id: Target diagram identifier
        query: Natural language search query
        method: Search method (auto, vector, cypher, hybrid)
        top_k: Number of results to return
        min_similarity: Minimum similarity threshold
        include_explanation: Whether to generate explanation
        detailed_explanation: Whether to use detailed explanation
        include_visualization: Whether to generate visualization
        
    Returns:
        Search results
    """
    request = SearchRequestAPI(
        query=query,
        diagram_id=diagram_id,
        method=method,
        top_k=top_k,
        min_similarity=min_similarity,
        include_explanation=include_explanation,
        detailed_explanation=detailed_explanation,
        include_visualization=include_visualization
    )
    
    return await search_graph(request)


@router.post("/visualize")
async def create_visualization(
    diagram_id: str,
    query: str,
    viz_request: VisualizationRequestAPI = None
):
    """
    Search and visualize graph results.
    
    Args:
        diagram_id: Target diagram identifier
        query: Search query
        viz_request: Visualization parameters
        
    Returns:
        Search results with visualization
    """
    try:
        if viz_request is None:
            viz_request = VisualizationRequestAPI()
        
        # Create search request with visualization enabled
        search_request = SearchRequest(
            query=query,
            diagram_id=diagram_id,
            method="auto",
            include_visualization=True
        )
        
        # Perform search with visualization
        result = graph_rag_service.search_and_visualize(search_request)
        
        response_data = {
            "query": result.request.query,
            "diagram_id": result.request.diagram_id,
            "success": result.success,
            "nodes": [node.dict() for node in result.nodes],
            "relationships": [rel.dict() for rel in result.relationships],
            "visualization_data": result.visualization_data,
            "error_message": result.error_message
        }
        
        if not result.success:
            return JSONResponse(
                status_code=422,
                content=ErrorResponse(
                    error="VisualizationError",
                    message=result.error_message or "Visualization failed",
                    timestamp=datetime.utcnow().isoformat() + "Z"
                ).dict()
            )
        
        return APIResponse(
            success=True,
            data=response_data,
            message="Visualization created successfully",
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=type(e).__name__,
                message=str(e),
                timestamp=datetime.utcnow().isoformat() + "Z"
            ).dict()
        )


@router.delete("/cache/{diagram_id}")
async def invalidate_cache(diagram_id: str):
    """
    Invalidate cached data for a diagram.
    
    Args:
        diagram_id: Diagram identifier
        
    Returns:
        Cache invalidation confirmation
    """
    try:
        graph_rag_service.invalidate_cache(diagram_id)
        
        return APIResponse(
            success=True,
            data={"diagram_id": diagram_id},
            message="Cache invalidated successfully",
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=type(e).__name__,
                message=str(e),
                timestamp=datetime.utcnow().isoformat() + "Z"
            ).dict()
        )


@router.get("/diagrams/{diagram_id}/stats")
async def get_diagram_statistics(diagram_id: str):
    """
    Get statistics for a diagram.
    
    Args:
        diagram_id: Diagram identifier
        
    Returns:
        Diagram statistics
    """
    try:
        stats = graph_rag_service.get_diagram_statistics(diagram_id)
        
        if not stats:
            raise HTTPException(
                status_code=404,
                detail=f"Diagram not found: {diagram_id}"
            )
        
        return APIResponse(
            success=True,
            data=stats,
            message="Diagram statistics retrieved successfully",
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=type(e).__name__,
                message=str(e),
                timestamp=datetime.utcnow().isoformat() + "Z"
            ).dict()
        )