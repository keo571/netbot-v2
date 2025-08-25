"""
Health check endpoints.
"""

from datetime import datetime
from fastapi import APIRouter, Depends

from ...shared import get_database, get_model_client, get_embedding_client, get_metrics
from ..models import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        System health status
    """
    services = {}
    
    try:
        # Check database connection
        db = get_database()
        db_stats = db.get_connection_stats()
        services["database"] = "connected" if db_stats else "disconnected"
    except Exception as e:
        services["database"] = f"error: {str(e)}"
    
    try:
        # Check model client
        model_client = get_model_client()
        services["model_client"] = "ready"
    except Exception as e:
        services["model_client"] = f"error: {str(e)}"
    
    try:
        # Check embedding client
        embedding_client = get_embedding_client()
        services["embedding_client"] = "ready"
    except Exception as e:
        services["embedding_client"] = f"error: {str(e)}"
    
    try:
        # Check metrics
        metrics = get_metrics()
        services["metrics"] = "collecting"
    except Exception as e:
        services["metrics"] = f"error: {str(e)}"
    
    # Determine overall status
    error_services = [name for name, status in services.items() if "error" in status]
    overall_status = "unhealthy" if error_services else "healthy"
    
    return HealthResponse(
        status=overall_status,
        services=services,
        timestamp=datetime.utcnow().isoformat() + "Z"
    )


@router.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check with metrics.
    
    Returns:
        Comprehensive system status
    """
    try:
        # Get basic health
        health = await health_check()
        
        # Add detailed metrics
        metrics = get_metrics()
        all_metrics = metrics.get_all_metrics()
        
        # Add database stats
        db = get_database()
        db_stats = db.get_connection_stats()
        
        return {
            "health": health.dict(),
            "metrics": all_metrics,
            "database_stats": db_stats,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        return {
            "error": f"Detailed health check failed: {str(e)}",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }