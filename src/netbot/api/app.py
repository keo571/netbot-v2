"""
FastAPI application factory for NetBot V2 API Gateway.

Creates a unified API for all NetBot services with proper middleware,
error handling, and monitoring.
"""

import time
from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from ..shared import get_logger, get_metrics, setup_logging, get_settings
from .models import APIResponse, ErrorResponse, HealthResponse
from .routers import diagram_processing, graph_rag, health, text_rag, rag_orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger = get_logger(__name__)
    
    # Startup
    logger.info("Starting NetBot V2 API Gateway...")
    setup_logging()
    
    # Initialize shared services
    try:
        settings = get_settings()
        logger.info(f"Loaded configuration: {settings.app_name}")
    except Exception as e:
        logger.error(f"Failed to initialize configuration: {e}")
        raise
    
    logger.info("NetBot V2 API Gateway started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down NetBot V2 API Gateway...")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI app instance
    """
    settings = get_settings()
    
    app = FastAPI(
        title="NetBot V2 API",
        description="""
        NetBot V2 API Gateway providing unified access to:
        - Diagram Processing: Convert images to knowledge graphs
        - Graph RAG: Semantic search and explanation generation
        - Text RAG: Document processing and semantic text search
        - Context Manager: Stateful conversation and user management
        - Visualization: Generate interactive graph visualizations
        """,
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        # Record metrics
        metrics = get_metrics()
        metrics.record_api_request(
            endpoint=request.url.path,
            method=request.method,
            duration_seconds=process_time,
            status_code=response.status_code
        )
        
        return response
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger = get_logger(__name__)
        logger.error(f"Unhandled exception in {request.method} {request.url}: {exc}")
        
        error_response = ErrorResponse(
            error=type(exc).__name__,
            message=str(exc),
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response.dict()
        )
    
    # HTTP exception handler
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        error_response = ErrorResponse(
            error="HTTPException",
            message=exc.detail,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.dict()
        )
    
    # Include routers
    app.include_router(health.router, prefix="/api/v2", tags=["Health"])
    app.include_router(diagram_processing.router, prefix="/api/v2", tags=["Diagram Processing"])
    app.include_router(graph_rag.router, prefix="/api/v2", tags=["Graph RAG"])
    app.include_router(text_rag.router, prefix="/api/v2", tags=["Text RAG"])
    
    # Import and include context manager router
    from .routers import context_manager
    app.include_router(context_manager.router, prefix="/api/v2", tags=["Context Manager"])
    
    # Include RAG Orchestrator (API Gateway) - Main entry point
    app.include_router(rag_orchestrator.router, tags=["RAG Orchestrator"])
    
    @app.get("/", response_model=APIResponse)
    async def root():
        """Root endpoint with API information."""
        return APIResponse(
            success=True,
            data={
                "name": "NetBot V2 API",
                "version": "2.0.0",
                "description": "Unified API for diagram processing and hybrid RAG system",
                "docs": "/docs",
                "health": "/api/v2/health",
                "rag_gateway": "/api/v1/rag",
                "features": [
                    "Hybrid RAG (Text + Graph + Context)",
                    "Diagram-to-Graph Pipeline",
                    "Conversational AI",
                    "Reliability Assessment",
                    "Batch Processing"
                ]
            },
            message="NetBot V2 Hybrid RAG API is running",
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
    
    return app