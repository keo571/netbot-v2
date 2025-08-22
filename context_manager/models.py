"""
Data models for the Context Manager system.

Defines the core data structures for session state, conversation history,
and user preferences as outlined in the docs/architecture/context-manager.md document.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum
import logging


logger = logging.getLogger(__name__)


class ResponseStyle(str, Enum):
    """User's preferred response style."""
    CONCISE = "concise"
    DETAILED = "detailed"
    BALANCED = "balanced"


class ExpertiseLevel(str, Enum):
    """User's expertise level for content personalization."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"


class QueryIntent(str, Enum):
    """Classification of user's query intent."""
    CLARIFICATION = "clarification"
    NEW_QUERY = "new_query"
    ELABORATION = "elaboration"
    FOLLOW_UP = "follow_up"


class UserFeedback(str, Enum):
    """User feedback on response quality."""
    THUMB_UP = "thumb_up"
    THUMB_DOWN = "thumb_down"
    NONE = "none"


@dataclass
class ConversationExchange:
    """A single exchange in a conversation."""
    query_text: str
    llm_response: str
    retrieved_context: Dict[str, Any]
    timestamp: datetime
    user_feedback: UserFeedback = UserFeedback.NONE
    diagram_ids_used: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "query_text": self.query_text,
            "llm_response": self.llm_response,
            "retrieved_context": self.retrieved_context,
            "timestamp": self.timestamp.isoformat(),
            "user_feedback": self.user_feedback.value,
            "diagram_ids_used": self.diagram_ids_used
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationExchange":
        """Create from dictionary."""
        return cls(
            query_text=data["query_text"],
            llm_response=data["llm_response"],
            retrieved_context=data["retrieved_context"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            user_feedback=UserFeedback(data.get("user_feedback", UserFeedback.NONE.value)),
            diagram_ids_used=data.get("diagram_ids_used", [])
        )


@dataclass
class SessionState:
    """
    Short-term memory for a single conversation session.
    Stored in fast cache (Redis) with TTL.
    """
    session_id: str
    last_retrieved_diagram_ids: List[str] = field(default_factory=list)
    active_entities: List[str] = field(default_factory=list)
    last_query_intent: Optional[QueryIntent] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        return {
            "session_id": self.session_id,
            "last_retrieved_diagram_ids": self.last_retrieved_diagram_ids,
            "active_entities": self.active_entities,
            "last_query_intent": self.last_query_intent.value if self.last_query_intent else None,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        """Create from Redis data."""
        return cls(
            session_id=data["session_id"],
            last_retrieved_diagram_ids=data.get("last_retrieved_diagram_ids", []),
            active_entities=data.get("active_entities", []),
            last_query_intent=QueryIntent(data["last_query_intent"]) if data.get("last_query_intent") else None,
            timestamp=datetime.fromisoformat(data["timestamp"])
        )
    
    def update_entities(self, new_entities: List[str], max_entities: int = 10):
        """Update active entities, keeping only the most recent ones."""
        self.active_entities.extend(new_entities)
        # Keep only unique entities and limit to max_entities
        self.active_entities = list(dict.fromkeys(self.active_entities))[-max_entities:]
        self.timestamp = datetime.now()
    
    def update_diagram_ids(self, diagram_ids: List[str], max_ids: int = 5):
        """Update last retrieved diagram IDs."""
        self.last_retrieved_diagram_ids.extend(diagram_ids)
        # Keep only unique IDs and limit to max_ids  
        self.last_retrieved_diagram_ids = list(dict.fromkeys(self.last_retrieved_diagram_ids))[-max_ids:]
        self.timestamp = datetime.now()


@dataclass
class ConversationHistory:
    """
    Long-term conversation history for a user.
    Stored persistently in document database (MongoDB).
    """
    user_id: str
    conversation_log: List[ConversationExchange] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_exchange(self, exchange: ConversationExchange):
        """Add a new conversation exchange."""
        self.conversation_log.append(exchange)
        self.updated_at = datetime.now()
    
    def get_recent_exchanges(self, n: int = 3) -> List[ConversationExchange]:
        """Get the N most recent exchanges."""
        return self.conversation_log[-n:] if self.conversation_log else []
    
    def get_exchanges_by_timeframe(self, hours: int = 24) -> List[ConversationExchange]:
        """Get exchanges from the last N hours."""
        cutoff = datetime.now().replace(hour=datetime.now().hour - hours)
        return [ex for ex in self.conversation_log if ex.timestamp >= cutoff]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB storage."""
        return {
            "user_id": self.user_id,
            "conversation_log": [ex.to_dict() for ex in self.conversation_log],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationHistory":
        """Create from MongoDB data."""
        return cls(
            user_id=data["user_id"],
            conversation_log=[ConversationExchange.from_dict(ex) for ex in data.get("conversation_log", [])],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )


@dataclass
class ExpertiseSignals:
    """Signals used to learn user expertise level."""
    technical_term_count: int = 0
    clarification_request_count: int = 0
    complex_query_count: int = 0
    follow_up_question_count: int = 0
    total_queries: int = 0
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "technical_term_count": self.technical_term_count,
            "clarification_request_count": self.clarification_request_count,
            "complex_query_count": self.complex_query_count,
            "follow_up_question_count": self.follow_up_question_count,
            "total_queries": self.total_queries
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "ExpertiseSignals":
        return cls(
            technical_term_count=data.get("technical_term_count", 0),
            clarification_request_count=data.get("clarification_request_count", 0),
            complex_query_count=data.get("complex_query_count", 0),
            follow_up_question_count=data.get("follow_up_question_count", 0),
            total_queries=data.get("total_queries", 0)
        )


@dataclass  
class UserPreferences:
    """
    User preferences and learned behaviors.
    Stored persistently in SQL or key-value database.
    """
    user_id: str
    response_style: ResponseStyle = ResponseStyle.BALANCED
    expertise_level: ExpertiseLevel = ExpertiseLevel.INTERMEDIATE
    preferred_formats: List[str] = field(default_factory=lambda: ['mermaid_diagram', 'bullet_points', 'paragraph'])
    topic_interest_profile: List[str] = field(default_factory=list)
    expertise_signals: ExpertiseSignals = field(default_factory=ExpertiseSignals)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def update_topic_interests(self, topics: List[str], max_topics: int = 20):
        """Update topic interest profile with new topics."""
        self.topic_interest_profile.extend(topics)
        # Keep only unique topics and limit count
        self.topic_interest_profile = list(dict.fromkeys(self.topic_interest_profile))[-max_topics:]
        self.updated_at = datetime.now()
    
    def adjust_response_style_score(self, feedback_signal: float):
        """
        Implicitly adjust response style based on user behavior.
        feedback_signal: -1.0 to 1.0, where positive means user wants more detail.
        """
        # Simple adjustment logic - can be made more sophisticated
        if feedback_signal > 0.5 and self.response_style == ResponseStyle.CONCISE:
            self.response_style = ResponseStyle.BALANCED
        elif feedback_signal > 0.8 and self.response_style == ResponseStyle.BALANCED:
            self.response_style = ResponseStyle.DETAILED
        elif feedback_signal < -0.5 and self.response_style == ResponseStyle.DETAILED:
            self.response_style = ResponseStyle.BALANCED
        elif feedback_signal < -0.8 and self.response_style == ResponseStyle.BALANCED:
            self.response_style = ResponseStyle.CONCISE
        
        self.updated_at = datetime.now()
    
    def update_expertise_signals(self, 
                                 has_technical_terms: bool = False,
                                 is_clarification_request: bool = False,
                                 is_complex_query: bool = False,
                                 is_follow_up_question: bool = False):
        """
        Update expertise signals based on query analysis.
        
        Args:
            has_technical_terms: Query contains technical terminology
            is_clarification_request: User asking for explanation/clarification
            is_complex_query: Query shows deep understanding
            is_follow_up_question: User asking follow-up to previous response
        """
        self.expertise_signals.total_queries += 1
        
        if has_technical_terms:
            self.expertise_signals.technical_term_count += 1
        
        if is_clarification_request:
            self.expertise_signals.clarification_request_count += 1
        
        if is_complex_query:
            self.expertise_signals.complex_query_count += 1
        
        if is_follow_up_question:
            self.expertise_signals.follow_up_question_count += 1
        
        # Adjust expertise level based on signals
        self._adjust_expertise_level()
        self.updated_at = datetime.now()
    
    def _adjust_expertise_level(self):
        """
        Adjust expertise level based on accumulated signals.
        Uses simple thresholds and ratios to determine appropriate level.
        """
        signals = self.expertise_signals
        
        # Need minimum queries before adjusting (avoid premature classification)
        if signals.total_queries < 5:
            return
        
        # Calculate ratios
        technical_ratio = signals.technical_term_count / signals.total_queries
        clarification_ratio = signals.clarification_request_count / signals.total_queries
        complex_ratio = signals.complex_query_count / signals.total_queries
        
        # Expert indicators: high technical terms, complex queries, low clarification requests
        expert_score = technical_ratio + complex_ratio - (clarification_ratio * 0.5)
        
        # Beginner indicators: high clarification requests, low technical terms
        beginner_score = clarification_ratio - technical_ratio - complex_ratio
        
        # Adjust expertise level based on scores
        current_level = self.expertise_level
        
        # Only adjust if we have strong signals (avoid constant fluctuation)
        if expert_score >= 0.4 and current_level != ExpertiseLevel.EXPERT:
            if current_level == ExpertiseLevel.BEGINNER and expert_score >= 0.6:
                self.expertise_level = ExpertiseLevel.INTERMEDIATE
            elif current_level == ExpertiseLevel.INTERMEDIATE and expert_score >= 0.5:
                self.expertise_level = ExpertiseLevel.EXPERT
        
        elif beginner_score >= 0.3 and current_level != ExpertiseLevel.BEGINNER:
            if current_level == ExpertiseLevel.EXPERT and beginner_score >= 0.5:
                self.expertise_level = ExpertiseLevel.INTERMEDIATE
            elif current_level == ExpertiseLevel.INTERMEDIATE and beginner_score >= 0.4:
                self.expertise_level = ExpertiseLevel.BEGINNER
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "user_id": self.user_id,
            "response_style": self.response_style.value,
            "expertise_level": self.expertise_level.value,
            "preferred_formats": self.preferred_formats,
            "topic_interest_profile": self.topic_interest_profile,
            "expertise_signals": self.expertise_signals.to_dict(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPreferences":
        """Create from database data."""
        return cls(
            user_id=data["user_id"],
            response_style=ResponseStyle(data.get("response_style", ResponseStyle.BALANCED.value)),
            expertise_level=ExpertiseLevel(data.get("expertise_level", ExpertiseLevel.INTERMEDIATE.value)),
            preferred_formats=data.get("preferred_formats", ['mermaid_diagram', 'bullet_points', 'paragraph']),
            topic_interest_profile=data.get("topic_interest_profile", []),
            expertise_signals=ExpertiseSignals.from_dict(data.get("expertise_signals", {})),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )