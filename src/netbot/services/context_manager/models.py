"""
Data models for Context Manager service.

Provides structured representations for sessions, messages, users, and configuration.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from enum import Enum
from pydantic import Field, validator

from ...shared.models.base import BaseModel, TimestampMixin


class MessageType(str, Enum):
    """Types of messages in conversation."""
    USER_QUERY = "user_query"
    SYSTEM_RESPONSE = "system_response"
    SYSTEM_INFO = "system_info"
    ERROR = "error"


class ResponseStyle(str, Enum):
    """User response style preferences."""
    BRIEF = "brief"
    DETAILED = "detailed"
    TECHNICAL = "technical"
    CONVERSATIONAL = "conversational"


class ExpertiseLevel(str, Enum):
    """User expertise levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class SessionStatus(str, Enum):
    """Session status values."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    ARCHIVED = "archived"


class Message(BaseModel, TimestampMixin):
    """
    Represents a single message in a conversation.
    """
    
    message_id: str = Field(..., description="Unique message identifier")
    session_id: str = Field(..., description="Associated session ID")
    content: str = Field(..., description="Message content")
    message_type: MessageType = Field(..., description="Type of message")
    
    # Context and metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional message metadata")
    diagram_context: Optional[str] = Field(default=None, description="Associated diagram ID")
    entities_mentioned: List[str] = Field(default_factory=list, description="Entities referenced")
    intent: Optional[str] = Field(default=None, description="Detected user intent")
    
    # Quality metrics
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Message confidence")
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Relevance to conversation")
    
    @validator('content')
    def validate_content(cls, v):
        """Validate message content."""
        if not v or not v.strip():
            raise ValueError("Message content cannot be empty")
        return v.strip()


class Session(BaseModel, TimestampMixin):
    """
    Represents a conversation session with context and state.
    """
    
    session_id: str = Field(..., description="Unique session identifier")
    user_id: str = Field(..., description="Associated user ID")
    diagram_id: Optional[str] = Field(default=None, description="Current active diagram")
    status: SessionStatus = Field(default=SessionStatus.ACTIVE, description="Session status")
    
    # Conversation state
    active_entities: Set[str] = Field(default_factory=set, description="Currently discussed entities")
    last_query_intent: Optional[str] = Field(default=None, description="Last detected intent")
    conversation_topic: Optional[str] = Field(default=None, description="Main conversation topic")
    
    # Context tracking
    last_activity: datetime = Field(default_factory=datetime.utcnow, description="Last activity timestamp")
    message_count: int = Field(default=0, description="Total messages in session")
    
    # Session metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional session metadata")
    preferences_override: Dict[str, Any] = Field(default_factory=dict, description="Session-specific preferences")
    
    @property
    def is_active(self) -> bool:
        """Check if session is currently active."""
        return self.status == SessionStatus.ACTIVE
    
    @property
    def duration_minutes(self) -> float:
        """Get session duration in minutes."""
        return (datetime.utcnow() - self.created_at).total_seconds() / 60
    
    @property
    def inactive_minutes(self) -> float:
        """Get minutes since last activity."""
        return (datetime.utcnow() - self.last_activity).total_seconds() / 60
    
    def should_expire(self, timeout_minutes: int = 30) -> bool:
        """Check if session should expire due to inactivity."""
        return self.inactive_minutes > timeout_minutes
    
    def add_entity(self, entity: str) -> None:
        """Add an entity to active discussion."""
        self.active_entities.add(entity)
    
    def remove_entity(self, entity: str) -> None:
        """Remove an entity from active discussion."""
        self.active_entities.discard(entity)
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()
    
    class Config:
        # Allow set type for active_entities
        arbitrary_types_allowed = True
        json_encoders = {
            set: list  # Convert sets to lists for JSON serialization
        }


class User(BaseModel, TimestampMixin):
    """
    Represents a user with preferences and conversation history.
    """
    
    user_id: str = Field(..., description="Unique user identifier")
    
    # User preferences
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    response_style: ResponseStyle = Field(default=ResponseStyle.DETAILED, description="Preferred response style")
    expertise_level: ExpertiseLevel = Field(default=ExpertiseLevel.INTERMEDIATE, description="User expertise level")
    
    # Learning and interests
    topic_interests: List[str] = Field(default_factory=list, description="User's topic interests")
    frequent_entities: Dict[str, int] = Field(default_factory=dict, description="Frequently mentioned entities")
    learned_preferences: Dict[str, Any] = Field(default_factory=dict, description="Implicitly learned preferences")
    
    # Activity tracking
    total_sessions: int = Field(default=0, description="Total number of sessions")
    total_messages: int = Field(default=0, description="Total messages sent")
    last_seen: datetime = Field(default_factory=datetime.utcnow, description="Last activity timestamp")
    
    # User metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional user metadata")
    
    @property
    def is_active_user(self) -> bool:
        """Check if user has been active recently."""
        days_since_last_seen = (datetime.utcnow() - self.last_seen).days
        return days_since_last_seen < 7
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get user preference with fallback to learned preferences."""
        # Check explicit preferences first
        if key in self.preferences:
            return self.preferences[key]
        
        # Check learned preferences
        if key in self.learned_preferences:
            return self.learned_preferences[key]
        
        return default
    
    def update_entity_frequency(self, entity: str) -> None:
        """Update frequency count for an entity."""
        self.frequent_entities[entity] = self.frequent_entities.get(entity, 0) + 1
    
    def get_top_entities(self, limit: int = 5) -> List[tuple]:
        """Get most frequently mentioned entities."""
        sorted_entities = sorted(
            self.frequent_entities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_entities[:limit]
    
    def update_activity(self) -> None:
        """Update last seen timestamp."""
        self.last_seen = datetime.utcnow()


class ContextConfig(BaseModel):
    """
    Configuration for Context Manager service.
    """
    
    # Session management
    session_timeout_minutes: int = Field(default=30, description="Session timeout in minutes")
    max_conversation_history: int = Field(default=100, description="Maximum messages to retain")
    auto_cleanup_enabled: bool = Field(default=True, description="Enable automatic session cleanup")
    
    # Learning settings
    enable_implicit_learning: bool = Field(default=True, description="Enable learning from user behavior")
    learning_threshold: int = Field(default=5, description="Minimum interactions before learning")
    
    # Storage settings
    storage_backend_type: str = Field(default="memory", description="Storage backend type")
    database_url: Optional[str] = Field(default=None, description="Database connection URL")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    
    # Processing settings
    context_window_size: int = Field(default=4000, description="Context window size for prompts")
    relevance_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Relevance threshold")
    max_entities_tracked: int = Field(default=20, description="Maximum entities to track per session")
    
    # Query enhancement
    enable_query_rewriting: bool = Field(default=True, description="Enable query enhancement")
    enable_pronoun_resolution: bool = Field(default=True, description="Enable pronoun resolution")
    enable_context_injection: bool = Field(default=True, description="Enable context injection")
    
    # Response filtering
    enable_response_filtering: bool = Field(default=True, description="Enable response filtering")
    personalization_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Personalization weight")
    
    @validator('session_timeout_minutes')
    def validate_timeout(cls, v):
        """Validate session timeout."""
        if v <= 0:
            raise ValueError("Session timeout must be positive")
        return v
    
    @validator('max_conversation_history')
    def validate_history_size(cls, v):
        """Validate conversation history size."""
        if v <= 0:
            raise ValueError("Conversation history size must be positive")
        return v


class ConversationSummary(BaseModel):
    """
    Summary of a conversation or session.
    """
    
    session_id: str = Field(..., description="Session identifier")
    user_id: str = Field(..., description="User identifier")
    
    # Summary content
    summary_text: str = Field(..., description="Generated conversation summary")
    key_topics: List[str] = Field(default_factory=list, description="Main topics discussed")
    key_entities: List[str] = Field(default_factory=list, description="Important entities mentioned")
    
    # Metadata
    message_count: int = Field(default=0, description="Number of messages summarized")
    summary_timestamp: datetime = Field(default_factory=datetime.utcnow, description="When summary was created")
    summary_method: str = Field(default="automatic", description="How summary was generated")
    
    # Quality metrics
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Summary completeness")
    coherence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Summary coherence")


class ContextQuery(BaseModel):
    """
    Represents an enhanced query with context.
    """
    
    original_query: str = Field(..., description="Original user query")
    enhanced_query: str = Field(..., description="Context-enhanced query")
    session_id: str = Field(..., description="Associated session ID")
    
    # Enhancement metadata
    enhancements_applied: List[str] = Field(default_factory=list, description="Types of enhancements applied")
    entities_resolved: Dict[str, str] = Field(default_factory=dict, description="Entity resolutions applied")
    context_added: List[str] = Field(default_factory=list, description="Context elements added")
    
    # Quality metrics
    enhancement_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Enhancement confidence")
    context_relevance: float = Field(default=0.0, ge=0.0, le=1.0, description="Context relevance score")
    
    @property
    def was_enhanced(self) -> bool:
        """Check if query was actually enhanced."""
        return self.original_query != self.enhanced_query
    
    @property
    def enhancement_summary(self) -> str:
        """Get summary of enhancements applied."""
        if not self.enhancements_applied:
            return "No enhancements applied"
        return f"Applied: {', '.join(self.enhancements_applied)}"