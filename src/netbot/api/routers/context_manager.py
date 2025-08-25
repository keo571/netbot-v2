"""
Context Manager API endpoints.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ...services.context_manager import ContextManager
from ...services.context_manager.models import ContextConfig
from ..models import APIResponse, ErrorResponse

router = APIRouter()

# Initialize context manager service
context_manager = ContextManager.create_with_database_storage()


class SessionStartRequest(BaseModel):
    """Request to start a new session."""
    user_id: str
    diagram_id: Optional[str] = None
    preferences: Dict[str, Any] = {}


class QueryEnhanceRequest(BaseModel):
    """Request to enhance a query with context."""
    session_id: str
    query: str


class SessionUpdateRequest(BaseModel):
    """Request to update session with activity."""
    session_id: str
    query: Optional[str] = None
    response: Optional[str] = None
    retrieved_context: List[Dict[str, Any]] = []
    user_feedback: Dict[str, Any] = {}


class UserPreferencesRequest(BaseModel):
    """Request to update user preferences."""
    user_id: str
    preferences: Dict[str, Any]


class ResultFilterRequest(BaseModel):
    """Request to filter search results."""
    session_id: str
    search_results: List[Dict[str, Any]]
    relevance_threshold: float = 0.6


@router.post("/context/sessions")
async def start_session(request: SessionStartRequest):
    """
    Start a new conversation session.
    
    Args:
        request: Session start parameters
        
    Returns:
        New session information
    """
    try:
        session = context_manager.start_session(
            user_id=request.user_id,
            diagram_id=request.diagram_id,
            user_preferences=request.preferences
        )
        
        return APIResponse(
            success=True,
            data={
                "session_id": session.session_id,
                "user_id": session.user_id,
                "diagram_id": session.diagram_id,
                "status": session.status.value,
                "created_at": session.created_at.isoformat()
            },
            message="Session started successfully",
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


@router.get("/context/sessions/{session_id}")
async def get_session(session_id: str):
    """
    Get session information.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Session details
    """
    try:
        session = context_manager.get_session(session_id)
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {session_id}"
            )
        
        return APIResponse(
            success=True,
            data={
                "session_id": session.session_id,
                "user_id": session.user_id,
                "diagram_id": session.diagram_id,
                "status": session.status.value,
                "active_entities": list(session.active_entities),
                "message_count": session.message_count,
                "duration_minutes": session.duration_minutes,
                "last_activity": session.last_activity.isoformat(),
                "created_at": session.created_at.isoformat()
            },
            message="Session retrieved successfully",
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


@router.post("/context/sessions/{session_id}/end")
async def end_session(session_id: str):
    """
    End a session gracefully.
    
    Args:
        session_id: Session to end
        
    Returns:
        Confirmation of session end
    """
    try:
        success = context_manager.end_session(session_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found or already ended: {session_id}"
            )
        
        return APIResponse(
            success=True,
            data={"session_id": session_id, "ended": True},
            message="Session ended successfully",
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


@router.post("/context/enhance")
async def enhance_query(request: QueryEnhanceRequest):
    """
    Enhance a query with contextual information.
    
    Args:
        request: Query enhancement parameters
        
    Returns:
        Enhanced query with context metadata
    """
    try:
        enhanced_query = context_manager.enhance_query(
            session_id=request.session_id,
            raw_query=request.query
        )
        
        return APIResponse(
            success=True,
            data={
                "original_query": enhanced_query.original_query,
                "enhanced_query": enhanced_query.enhanced_query,
                "session_id": enhanced_query.session_id,
                "was_enhanced": enhanced_query.was_enhanced,
                "enhancements_applied": enhanced_query.enhancements_applied,
                "entities_resolved": enhanced_query.entities_resolved,
                "context_added": enhanced_query.context_added,
                "enhancement_confidence": enhanced_query.enhancement_confidence,
                "context_relevance": enhanced_query.context_relevance
            },
            message="Query enhanced successfully",
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


@router.post("/context/sessions/update")
async def update_session(request: SessionUpdateRequest):
    """
    Update session with conversation activity.
    
    Args:
        request: Session update parameters
        
    Returns:
        Updated session information
    """
    try:
        session = context_manager.update_session(
            session_id=request.session_id,
            query=request.query,
            response=request.response,
            retrieved_context=request.retrieved_context,
            user_feedback=request.user_feedback
        )
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {request.session_id}"
            )
        
        return APIResponse(
            success=True,
            data={
                "session_id": session.session_id,
                "message_count": session.message_count,
                "active_entities": list(session.active_entities),
                "last_activity": session.last_activity.isoformat(),
                "updated": True
            },
            message="Session updated successfully",
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


@router.post("/context/filter")
async def filter_results(request: ResultFilterRequest):
    """
    Filter and re-rank search results based on context.
    
    Args:
        request: Result filtering parameters
        
    Returns:
        Filtered and contextually re-ranked results
    """
    try:
        filtered_results = context_manager.filter_results(
            session_id=request.session_id,
            search_results=request.search_results,
            relevance_threshold=request.relevance_threshold
        )
        
        return APIResponse(
            success=True,
            data={
                "original_count": len(request.search_results),
                "filtered_count": len(filtered_results),
                "filtering_ratio": len(filtered_results) / len(request.search_results) if request.search_results else 0,
                "filtered_results": filtered_results,
                "relevance_threshold": request.relevance_threshold
            },
            message="Results filtered successfully",
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


@router.post("/context/users/preferences")
async def update_user_preferences(request: UserPreferencesRequest):
    """
    Update user preferences.
    
    Args:
        request: User preference update parameters
        
    Returns:
        Updated user profile
    """
    try:
        user = context_manager.update_user_preferences(
            user_id=request.user_id,
            preferences=request.preferences
        )
        
        if not user:
            # Create user if doesn't exist
            user = context_manager.get_user(request.user_id)
        
        return APIResponse(
            success=True,
            data={
                "user_id": user.user_id,
                "response_style": user.response_style.value,
                "expertise_level": user.expertise_level.value,
                "preferences": user.preferences,
                "topic_interests": user.topic_interests,
                "last_seen": user.last_seen.isoformat()
            },
            message="User preferences updated successfully",
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


@router.get("/context/users/{user_id}")
async def get_user(user_id: str):
    """
    Get user profile and preferences.
    
    Args:
        user_id: User identifier
        
    Returns:
        User profile information
    """
    try:
        user = context_manager.get_user(user_id)
        
        return APIResponse(
            success=True,
            data={
                "user_id": user.user_id,
                "response_style": user.response_style.value,
                "expertise_level": user.expertise_level.value,
                "preferences": user.preferences,
                "topic_interests": user.topic_interests,
                "frequent_entities": user.frequent_entities,
                "total_sessions": user.total_sessions,
                "total_messages": user.total_messages,
                "last_seen": user.last_seen.isoformat(),
                "is_active": user.is_active_user,
                "created_at": user.created_at.isoformat()
            },
            message="User profile retrieved successfully",
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


@router.get("/context/users/{user_id}/sessions")
async def get_user_sessions(
    user_id: str,
    active_only: bool = Query(False, description="Return only active sessions")
):
    """
    Get sessions for a user.
    
    Args:
        user_id: User identifier
        active_only: Whether to return only active sessions
        
    Returns:
        List of user sessions
    """
    try:
        if active_only:
            sessions = context_manager.get_active_sessions(user_id)
        else:
            # Get user insights which includes session information
            insights = context_manager.get_user_insights(user_id)
            session_count = insights.get('session_count', 0)
            
            # For now, just get active sessions
            # In a full implementation, you'd get all sessions from storage
            sessions = context_manager.get_active_sessions(user_id)
        
        session_data = []
        for session in sessions:
            session_data.append({
                "session_id": session.session_id,
                "diagram_id": session.diagram_id,
                "status": session.status.value,
                "message_count": session.message_count,
                "duration_minutes": session.duration_minutes,
                "last_activity": session.last_activity.isoformat(),
                "created_at": session.created_at.isoformat()
            })
        
        return APIResponse(
            success=True,
            data={
                "user_id": user_id,
                "sessions": session_data,
                "session_count": len(session_data),
                "active_only": active_only
            },
            message="User sessions retrieved successfully",
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


@router.get("/context/sessions/{session_id}/history")
async def get_conversation_history(
    session_id: str,
    limit: int = Query(50, description="Maximum number of messages to return")
):
    """
    Get conversation history for a session.
    
    Args:
        session_id: Session identifier
        limit: Maximum number of messages
        
    Returns:
        Conversation history
    """
    try:
        messages = context_manager.get_conversation_history(session_id, limit)
        
        message_data = []
        for message in messages:
            message_data.append({
                "message_id": message.message_id,
                "content": message.content,
                "message_type": message.message_type.value,
                "entities_mentioned": message.entities_mentioned,
                "intent": message.intent,
                "confidence_score": message.confidence_score,
                "created_at": message.created_at.isoformat()
            })
        
        return APIResponse(
            success=True,
            data={
                "session_id": session_id,
                "messages": message_data,
                "message_count": len(message_data),
                "limit": limit
            },
            message="Conversation history retrieved successfully",
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


@router.post("/context/sessions/{session_id}/summarize")
async def summarize_conversation(session_id: str):
    """
    Generate a summary of the conversation.
    
    Args:
        session_id: Session to summarize
        
    Returns:
        Conversation summary
    """
    try:
        summary = context_manager.summarize_conversation(session_id)
        
        if not summary:
            raise HTTPException(
                status_code=404,
                detail=f"No conversation found for session: {session_id}"
            )
        
        return APIResponse(
            success=True,
            data=summary,
            message="Conversation summarized successfully",
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


@router.get("/context/analytics/sessions/{session_id}")
async def get_session_analytics(session_id: str):
    """
    Get analytics for a session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Session analytics and insights
    """
    try:
        analytics = context_manager.get_session_analytics(session_id)
        
        if not analytics:
            raise HTTPException(
                status_code=404,
                detail=f"No analytics found for session: {session_id}"
            )
        
        return APIResponse(
            success=True,
            data=analytics,
            message="Session analytics retrieved successfully", 
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


@router.get("/context/analytics/users/{user_id}")
async def get_user_insights(user_id: str):
    """
    Get behavioral insights for a user.
    
    Args:
        user_id: User identifier
        
    Returns:
        User behavioral insights and patterns
    """
    try:
        insights = context_manager.get_user_insights(user_id)
        
        return APIResponse(
            success=True,
            data=insights,
            message="User insights retrieved successfully",
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


@router.post("/context/maintenance/cleanup")
async def cleanup_expired_sessions():
    """
    Manually trigger cleanup of expired sessions.
    
    Returns:
        Number of sessions cleaned up
    """
    try:
        count = context_manager.cleanup_expired_sessions()
        
        return APIResponse(
            success=True,
            data={
                "cleaned_sessions": count,
                "timestamp": datetime.utcnow().isoformat()
            },
            message=f"Cleaned up {count} expired sessions",
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


@router.get("/context/stats")
async def get_service_stats():
    """
    Get Context Manager service statistics.
    
    Returns:
        Service health and statistics
    """
    try:
        stats = context_manager.get_service_stats()
        
        return APIResponse(
            success=True,
            data=stats,
            message="Service statistics retrieved successfully",
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