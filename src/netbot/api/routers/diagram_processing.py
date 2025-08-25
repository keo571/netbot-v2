"""
Diagram Processing API endpoints.
"""

from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

from ...services.diagram_processing import DiagramProcessingService
from ...services.diagram_processing.models import ProcessingRequest
from ..models import APIResponse, ErrorResponse

router = APIRouter()

# Initialize service
processing_service = DiagramProcessingService()


@router.post("/diagrams/process")
async def process_diagram(
    diagram_id: str = Form(..., description="Unique diagram identifier"),
    file: UploadFile = File(..., description="Diagram image file"),
    ocr_enabled: bool = Form(True, description="Enable OCR processing"),
    shape_detection: bool = Form(True, description="Enable shape detection"),
    store_in_database: bool = Form(True, description="Store results in Neo4j"),
    model_name: str = Form("gemini-2.0-flash-exp", description="AI model to use"),
    max_tokens: int = Form(8192, description="Maximum response tokens"),
    temperature: float = Form(0.1, description="AI generation temperature")
):
    """
    Process a diagram image into structured graph data.
    
    Args:
        diagram_id: Unique identifier for the diagram
        file: Uploaded image file
        ocr_enabled: Whether to enable OCR text extraction
        shape_detection: Whether to enable shape detection
        store_in_database: Whether to store results in Neo4j
        model_name: AI model to use for processing
        max_tokens: Maximum tokens for AI response
        temperature: Temperature for AI generation
        
    Returns:
        Processing results with extracted nodes and relationships
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file type
        allowed_types = {'image/jpeg', 'image/png', 'image/jpg', 'image/bmp', 'image/tiff'}
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type: {file.content_type}. Allowed types: {allowed_types}"
            )
        
        # Save uploaded file temporarily
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_path = tmp_file.name
        
        try:
            # Create processing request
            request = ProcessingRequest(
                image_path=temp_path,
                diagram_id=diagram_id,
                ocr_enabled=ocr_enabled,
                shape_detection=shape_detection,
                store_in_database=store_in_database,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Process the diagram
            result = processing_service.process_diagram(request)
            
            # Return results
            response_data = {
                "diagram_id": result.request.diagram_id,
                "success": result.success,
                "nodes": [node.dict() for node in result.nodes],
                "relationships": [rel.dict() for rel in result.relationships],
                "node_count": result.node_count,
                "relationship_count": result.relationship_count,
                "confidence_score": result.confidence_score,
                "processing_time_seconds": result.total_duration_seconds,
                "error_message": result.error_message
            }
            
            if not result.success:
                return JSONResponse(
                    status_code=422,
                    content=ErrorResponse(
                        error="ProcessingError",
                        message=result.error_message or "Processing failed",
                        timestamp=datetime.utcnow().isoformat() + "Z"
                    ).dict()
                )
            
            return APIResponse(
                success=True,
                data=response_data,
                message="Diagram processed successfully",
                timestamp=datetime.utcnow().isoformat() + "Z"
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
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


@router.get("/diagrams/{diagram_id}/stats")
async def get_diagram_stats(diagram_id: str):
    """
    Get statistics for a processed diagram.
    
    Args:
        diagram_id: Diagram identifier
        
    Returns:
        Diagram statistics
    """
    try:
        stats = processing_service.get_diagram_stats(diagram_id)
        
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


@router.get("/diagrams/history")
async def get_processing_history(limit: int = 10):
    """
    Get recent diagram processing history.
    
    Args:
        limit: Maximum number of results to return
        
    Returns:
        List of recently processed diagrams
    """
    try:
        history = processing_service.get_processing_history(limit)
        
        return APIResponse(
            success=True,
            data={
                "history": history,
                "count": len(history)
            },
            message="Processing history retrieved successfully",
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