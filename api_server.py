#!/usr/bin/env python3
"""
FastAPI server for netbot-v2 with role-based access control.

Public endpoints:
- /chat - Search existing diagrams
- /diagrams - List available diagrams

Admin endpoints:
- /admin/upload-diagram - Process new diagrams
- /admin/diagrams/{diagram_id} - Delete single diagram
- /admin/diagrams - Bulk delete multiple diagrams
- /admin/bulk-upload - Bulk process directory
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid
from datetime import datetime, timedelta, timezone
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import jwt
from dotenv import load_dotenv

from client import NetBot

# Load environment variables
load_dotenv()

# Track which diagrams have already sent visualizations
sent_visualizations = set()

# Configuration
SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-this-in-production')
ADMIN_API_KEY = os.getenv('ADMIN_API_KEY', 'admin-secret-key-change-this')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI(
    title="NetBot-v2 API",
    description="AI-powered diagram analysis with role-based access",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cache for pre-warmed data
_cached_netbot = None
_embeddings_cache_warmed = False


# Security
security = HTTPBearer(auto_error=False)

# Pydantic models

class NodeResponse(BaseModel):
    id: str
    label: str
    type: str
    properties: Dict[str, Any] = {}

class RelationshipResponse(BaseModel):
    id: str
    source_id: str
    target_id: str
    type: str
    properties: Dict[str, Any] = {}

class SearchResults(BaseModel):
    nodes: List[NodeResponse]
    relationships: List[RelationshipResponse]

class ChatResponse(BaseModel):
    response: str
    results: Optional[SearchResults] = None
    explanation: Optional[str] = None
    visualization_path: Optional[str] = None
    diagram_id: str
    query: str

class DiagramInfo(BaseModel):
    diagram_id: str
    node_count: int = 0
    relationship_count: int = 0

class UploadResponse(BaseModel):
    message: str
    diagram_id: str
    node_count: int
    relationship_count: int

class ChatRequest(BaseModel):
    message: str = Field(..., description="Natural language query")
    diagram_id: str = Field(..., description="Diagram ID to search")
    method: str = Field("auto", description="Search method: 'vector', 'cypher', or 'auto'")
    explanation_detail: str = Field("basic", description="Explanation level: 'none', 'basic', or 'detailed'")

class AdminLoginRequest(BaseModel):
    api_key: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str

class BulkDeleteRequest(BaseModel):
    diagram_ids: List[str]

class BulkDeleteResponse(BaseModel):
    message: str
    deleted_diagrams: int
    failed_deletions: List[str]
    deleted_files: List[str]


# Utility functions
def transform_to_api_results(raw_results: Dict[str, Any]) -> Optional[SearchResults]:
    """Transform internal search results to clean API response format"""
    if not raw_results or not raw_results.get('nodes'):
        return None
    
    # Transform nodes
    api_nodes = [
        NodeResponse(
            id=getattr(node, 'id', ''),
            label=getattr(node, 'label', ''),
            type=getattr(node, 'type', ''),
            properties=getattr(node, 'properties', {})
        )
        for node in raw_results.get('nodes', [])
    ]
    
    # Transform relationships
    api_relationships = [
        RelationshipResponse(
            id=getattr(rel, 'id', ''),
            source_id=getattr(rel, 'source_id', ''),
            target_id=getattr(rel, 'target_id', ''),
            type=getattr(rel, 'type', ''),
            properties=getattr(rel, 'properties', {})
        )
        for rel in raw_results.get('relationships', [])
    ]
    
    return SearchResults(nodes=api_nodes, relationships=api_relationships)

# Authentication functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_admin_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> bool:
    """Verify admin JWT token"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        role: str = payload.get("role")
        if role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        return True
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except (jwt.PyJWTError, jwt.DecodeError, jwt.ExpiredSignatureError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Utility functions
def get_netbot() -> NetBot:
    """Get cached NetBot instance with error handling"""
    global _cached_netbot, _embeddings_cache_warmed
    
    if _cached_netbot is None:
        try:
            _cached_netbot = NetBot()
            print("🚀 NetBot instance cached globally")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize NetBot: {str(e)}"
            )
    
    # Pre-warm embeddings cache on first use
    if not _embeddings_cache_warmed:
        try:
            asyncio.create_task(_prewarm_embeddings_cache())
            _embeddings_cache_warmed = True
        except Exception as e:
            print(f"⚠️ Failed to start cache prewarming: {e}")
    
    return _cached_netbot

async def _prewarm_embeddings_cache():
    """Pre-warm embeddings cache for known diagrams"""
    try:
        print("🔥 Pre-warming embeddings cache...")
        
        def _do_prewarm():
            try:
                # Get list of diagrams with embeddings
                from graph_rag.database.connection import Neo4jConnection
                db = Neo4jConnection(
                    uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
                    user=os.getenv('NEO4J_USER', 'neo4j'),
                    password=os.getenv('NEO4J_PASSWORD')
                )
                
                query = """
                MATCH (n)
                WHERE n.diagram_id IS NOT NULL AND n.embedding IS NOT NULL
                RETURN DISTINCT n.diagram_id as diagram_id
                LIMIT 5
                """
                
                with db.get_session() as session:
                    results = session.run(query)
                    diagram_ids = [record['diagram_id'] for record in results]
                
                db.close()
                
                if diagram_ids:
                    print(f"🔥 Pre-warming cache for {len(diagram_ids)} diagrams...")
                    # Pre-warm the first diagram's cache
                    for diagram_id in diagram_ids[:1]:  # Just warm the first one for now
                        try:
                            netbot = _cached_netbot
                            # Trigger cache loading
                            netbot.search("test", diagram_id, method="vector", top_k=1)
                            print(f"✅ Pre-warmed cache for {diagram_id}")
                            break  # Only pre-warm one diagram for speed
                        except Exception as e:
                            print(f"⚠️ Failed to pre-warm {diagram_id}: {e}")
                else:
                    print("📋 No diagrams with embeddings found for pre-warming")
                    
            except Exception as e:
                print(f"⚠️ Cache pre-warming failed: {e}")
        
        # Run in thread pool to avoid blocking startup
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            await loop.run_in_executor(executor, _do_prewarm)
            
    except Exception as e:
        print(f"⚠️ Cache pre-warming failed: {e}")

def save_uploaded_file(upload_file: UploadFile) -> str:
    """Save uploaded file to temporary location"""
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    file_extension = Path(upload_file.filename).suffix
    temp_filename = f"{uuid.uuid4()}{file_extension}"
    temp_path = temp_dir / temp_filename
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    return str(temp_path)

# Public endpoints
@app.get("/")
async def root():
    """API health check"""
    return {
        "message": "NetBot-v2 API is running",
        "version": "2.0.0",
        "endpoints": {
            "public": ["/chat", "/diagrams"],
            "admin": ["/admin/login", "/admin/upload-diagram", "/admin/diagrams/{id}", "/admin/diagrams", "/admin/bulk-upload"]
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Public endpoint: Search existing diagrams with natural language.
    
    Request body should contain:
    - message: Natural language search query (required)
    - diagram_id: Which diagram to search (required)  
    - method: Search method - "vector", "cypher", or "auto" (default: "auto")
    - explanation_detail: "none", "basic", or "detailed" (default: "basic")
    
    Returns relevant nodes, relationships, simple explanation, and embedded visualization.
    Optimized for speed with concise responses.
    """
    netbot = get_netbot()
    
    try:
        # Check if we need visualization for this diagram
        skip_visualization = request.diagram_id in sent_visualizations
        
        def run_search_operation():
            """Run search operation in thread pool"""
            if skip_visualization:
                # Use faster search-only method for repeated requests
                results = netbot.search(
                    query=request.message,
                    diagram_id=request.diagram_id,
                    method=request.method
                )
                # Add explanation if requested
                if request.explanation_detail != "none":
                    # Generate explanation directly
                    graph_rag = netbot.get_graph_rag()
                    try:
                        explanation = graph_rag.explain_subgraph(
                            results.get('nodes', []),
                            results.get('relationships', []),
                            request.message,
                            detailed=(request.explanation_detail == "detailed")
                        )
                        results['explanation'] = explanation
                    finally:
                        graph_rag.close()
                return results
            else:
                # First request - use full query_and_visualize
                return netbot.query_and_visualize(
                    query=request.message,
                    diagram_id=request.diagram_id,
                    backend="graphviz",
                    layout="dot",  # Better layout for hierarchical/star patterns
                    explanation_detail=request.explanation_detail,
                    method=request.method,
                    show_edge_properties=False  # Cleaner edge labels
                )
        
        # Run the search operation in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = await loop.run_in_executor(executor, run_search_operation)
        
        if not results.get('nodes'):
            return ChatResponse(
                response="No results found for your query",
                diagram_id=request.diagram_id,
                query=request.message,
                explanation="No relevant components found in the diagram for your query."
            )
        
        # Transform results to clean API format
        api_results = transform_to_api_results(results)
        
        # Handle visualization data
        visualization_data = None
        
        if skip_visualization:
            print(f"📋 Skipped visualization generation for {request.diagram_id}")
            visualization_data = None
        else:
            # First request - extract and send visualization
            image_base64 = results.get('image_path')  # Contains base64 data
            if image_base64:
                visualization_data = image_base64  # Already in base64 format
                sent_visualizations.add(request.diagram_id)
                print(f"📤 First-time visualization sent for {request.diagram_id}")
            else:
                print(f"⚠️ No visualization data available")
                visualization_data = None
        
        node_count = len(results.get('nodes', []))
        relationship_count = len(results.get('relationships', []))
        
        # Simple explanation or summary
        base_explanation = results.get('explanation') 
        if not base_explanation and request.explanation_detail != "none":
            # Generate simple explanation based on results
            node_types = list(set(getattr(node, 'type', 'Unknown') for node in results.get('nodes', [])))
            if node_types:
                base_explanation = f"Found network components of types: {', '.join(node_types[:3])}{'...' if len(node_types) > 3 else ''}"
            else:
                base_explanation = "Analysis completed successfully."
        elif not base_explanation:
            base_explanation = None
        
        # Summary with diagram_id in the main response (only if diagram_id is provided)
        if request.diagram_id and request.diagram_id.strip():
            summary = f"**Found {node_count} relevant components with {relationship_count} relationships in diagram: `{request.diagram_id}`**"
        else:
            summary = f"**Found {node_count} relevant components with {relationship_count} relationships**"
        
        return ChatResponse(
            response=summary,
            results=api_results,
            explanation=base_explanation,  # AI explanation in separate section
            visualization_path=visualization_data,  # Base64 data
            diagram_id=request.diagram_id,
            query=request.message
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@app.get("/diagrams", response_model=List[DiagramInfo])
async def list_diagrams():
    """
    Public endpoint: List all available diagrams for search.
    """
    try:
        # Get diagrams from database
        # Note: You'll need to implement this method in NetBot
        from graph_rag.database.connection import Neo4jConnection
        
        db = Neo4jConnection(
            uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            user=os.getenv('NEO4J_USER', 'neo4j'),
            password=os.getenv('NEO4J_PASSWORD')
        )
        
        # Get unique diagram IDs with stats
        query = """
        MATCH (n)
        WHERE n.diagram_id IS NOT NULL
        WITH n.diagram_id as diagram_id, COUNT(n) as node_count
        OPTIONAL MATCH (n {diagram_id: diagram_id})-[r]->(m {diagram_id: diagram_id})
        WITH diagram_id, node_count, COUNT(r) as rel_count
        RETURN diagram_id, node_count, rel_count
        ORDER BY diagram_id
        """
        
        with db.get_session() as session:
            results = session.run(query)
            diagrams = []
            
            for record in results:
                diagram_id = record['diagram_id']
                
                diagrams.append(DiagramInfo(
                    diagram_id=diagram_id,
                    node_count=record['node_count'],
                    relationship_count=record['rel_count']
                ))
        
        db.close()
        return diagrams
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list diagrams: {str(e)}"
        )

# Admin authentication endpoint
@app.post("/admin/login", response_model=TokenResponse)
async def admin_login(request: AdminLoginRequest):
    """
    Admin login endpoint: Exchange API key for JWT token
    """
    if request.api_key != ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"role": "admin"}, expires_delta=access_token_expires
    )
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer"
    )

# Admin-only endpoints
@app.post("/admin/upload-diagram", response_model=UploadResponse)
async def upload_diagram(
    file: UploadFile = File(...),
    _: bool = Depends(verify_admin_token)
):
    """
    Admin endpoint: Process and ingest new diagram image.
    Automatically generates embeddings for semantic search.
    
    The filename (without extension) becomes the diagram_id.
    Always processes uploaded files - if diagram exists, it will be overwritten.
    
    Supported formats: PNG, JPG, JPEG, BMP, TIFF, WebP
    """
    # Validate file type
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {file_extension} not supported. Allowed: {allowed_extensions}"
        )
    
    temp_path = None
    try:
        # Save uploaded file
        temp_path = save_uploaded_file(file)
        
        # Generate diagram_id from filename (always)
        import re
        base_name = os.path.splitext(file.filename)[0]
        # Normalize to lowercase for consistency and performance
        diagram_id = re.sub(r'[^\w\-_]', '_', base_name.lower())
        
        # Process diagram using NetBot client
        netbot = get_netbot()
        result = netbot.process_diagram(
            image_path=temp_path,
            diagram_id=diagram_id,
            output_dir="data/processed/diagrams", 
            force_reprocess=True
        )
        
        if result.get('status') != 'success':
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Diagram processing failed: {result.get('error')}"
            )
        
        # diagram_id is already set above, use the result data
        node_count = len(result.get('nodes', []))
        relationship_count = len(result.get('relationships', []))
        
        # Add embeddings for semantic search
        embedding_success = netbot.add_embeddings(diagram_id)
        
        return UploadResponse(
            message=f"Diagram processed successfully. Embeddings: {'✅' if embedding_success else '❌'}",
            diagram_id=diagram_id,
            node_count=node_count,
            relationship_count=relationship_count
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload processing failed: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/admin/bulk-upload")
async def bulk_upload_directory(
    directory_path: str,
    _: bool = Depends(verify_admin_token)
):
    """
    Admin endpoint: Bulk process all images in a directory.
    
    Processes all supported image files (PNG, JPG, JPEG, BMP, TIFF, WebP) in the specified directory.
    Each filename becomes a diagram_id. Existing diagrams will be overwritten.
    
    Example directory paths:
    - Linux/Mac: '/home/user/diagrams' or 'data/examples'  
    - Windows: 'C:\\Users\\user\\diagrams' or 'data\\examples'
    """
    if not os.path.exists(directory_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Directory not found: {directory_path}"
        )
    
    try:
        netbot = get_netbot()
        results = netbot.bulk_quickstart(directory_path)
        
        if results.get('error'):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Bulk processing failed: {results['error']}"
            )
        
        return {
            "message": "Bulk processing completed",
            "processed_diagrams": results.get('diagrams_processed', []),
            "embeddings_added": results.get('diagrams_with_embeddings', []),
            "summary": results.get('summary', {})
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Bulk upload failed: {str(e)}"
        )

@app.delete("/admin/diagrams/{diagram_id}")
async def delete_diagram(
    diagram_id: str,
    _: bool = Depends(verify_admin_token)
):
    """
    Admin endpoint: Delete single diagram from both Neo4j and filesystem.
    
    Removes all nodes, relationships, embeddings, and associated files for the specified diagram.
    This action cannot be undone.
    """
    try:
        from graph_rag.database.connection import Neo4jConnection
        
        db = Neo4jConnection(
            uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            user=os.getenv('NEO4J_USER', 'neo4j'),
            password=os.getenv('NEO4J_PASSWORD')
        )
        
        # Delete all nodes and relationships for this diagram
        delete_query = """
        MATCH (n {diagram_id: $diagram_id})
        OPTIONAL MATCH (n)-[r]-()
        DELETE r, n
        RETURN COUNT(n) as deleted_nodes
        """
        
        with db.get_session() as session:
            result = session.run(delete_query, {"diagram_id": diagram_id})
            record = result.single()
            deleted_count = record['deleted_nodes'] if record else 0
        
        db.close()
        
        if deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Diagram {diagram_id} not found"
            )
        
        # Also delete filesystem files (now always lowercase)
        deleted_files = []
        exact_dir_path = f"data/processed/diagrams/{diagram_id}"
        
        if os.path.exists(exact_dir_path):
            import shutil
            shutil.rmtree(exact_dir_path)
            deleted_files.append(exact_dir_path)
        
        return {
            "message": f"Diagram {diagram_id} deleted successfully",
            "deleted_nodes": deleted_count,
            "deleted_files": deleted_files
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deletion failed: {str(e)}"
        )

@app.delete("/admin/diagrams", response_model=BulkDeleteResponse)
async def bulk_delete_diagrams(
    request: BulkDeleteRequest,
    _: bool = Depends(verify_admin_token)
):
    """
    Admin endpoint: Bulk delete multiple diagrams from both Neo4j and filesystem.
    
    Args:
        request: BulkDeleteRequest with list of diagram_ids to delete
        
    Returns:
        BulkDeleteResponse with deletion summary
    """
    if not request.diagram_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No diagram IDs provided"
        )
    
    # Normalize all diagram_ids to lowercase for consistency
    normalized_ids = [diagram_id.lower() for diagram_id in request.diagram_ids]
    
    deleted_diagrams = 0
    failed_deletions = []
    all_deleted_files = []
    
    try:
        from graph_rag.database.connection import Neo4jConnection
        
        db = Neo4jConnection(
            uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            user=os.getenv('NEO4J_USER', 'neo4j'),
            password=os.getenv('NEO4J_PASSWORD')
        )
        
        for diagram_id in normalized_ids:
            try:
                # Delete from Neo4j
                delete_query = """
                MATCH (n {diagram_id: $diagram_id})
                OPTIONAL MATCH (n)-[r]-()
                DELETE r, n
                RETURN COUNT(n) as deleted_nodes
                """
                
                with db.get_session() as session:
                    result = session.run(delete_query, {"diagram_id": diagram_id})
                    record = result.single()
                    deleted_count = record['deleted_nodes'] if record else 0
                
                if deleted_count > 0:
                    deleted_diagrams += 1
                    
                    # Delete filesystem files
                    exact_dir_path = f"data/processed/diagrams/{diagram_id}"
                    if os.path.exists(exact_dir_path):
                        import shutil
                        shutil.rmtree(exact_dir_path)
                        all_deleted_files.append(exact_dir_path)
                else:
                    failed_deletions.append(f"{diagram_id}: not found in database")
                    
            except Exception as e:
                failed_deletions.append(f"{diagram_id}: {str(e)}")
        
        db.close()
        
        return BulkDeleteResponse(
            message=f"Bulk deletion completed: {deleted_diagrams} diagrams deleted, {len(failed_deletions)} failed",
            deleted_diagrams=deleted_diagrams,
            failed_deletions=failed_deletions,
            deleted_files=all_deleted_files
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Bulk deletion failed: {str(e)}"
        )

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "detail": "Check /docs for available endpoints"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "Please check server logs"}
    )

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting NetBot-v2 API server...")
    print("📖 API docs available at: http://localhost:8000/docs")
    print("🔒 Admin endpoints require authentication")
    uvicorn.run(app, host="0.0.0.0", port=8000)
