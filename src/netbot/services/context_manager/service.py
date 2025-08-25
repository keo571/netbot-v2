"""
Context Manager Service implementation.

Provides comprehensive context management for conversational AI systems
including session management, user preferences, and contextual processing.
"""

import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

from ...shared import (
    get_logger, get_metrics, get_model_client,
    ProcessingError, ConfigurationError
)
from .models import (
    Session, Message, User, ContextQuery, ConversationSummary,
    ContextConfig, SessionStatus, MessageType, ResponseStyle, ExpertiseLevel
)
from .storage.backends import StorageBackend, InMemoryStorage, ExternalStorage
from .core import PromptBuilder, QueryRewriter, RetrievalFilter


class ContextManagerService:
    """
    Comprehensive context management service.
    
    Provides session management, conversation history, user preferences,
    and contextual processing capabilities for conversational AI systems.
    """
    
    def __init__(self, 
                 config: ContextConfig = None,
                 storage_backend: StorageBackend = None):
        """
        Initialize the Context Manager service.
        
        Args:
            config: Service configuration
            storage_backend: Storage backend to use
        """
        self.logger = get_logger(__name__)
        self.metrics = get_metrics()
        self.model_client = get_model_client()
        
        # Configuration
        self.config = config or ContextConfig()
        
        # Storage backend
        if storage_backend:
            self.storage = storage_backend
        elif self.config.storage_backend_type == "external":
            self.storage = ExternalStorage(self.config.database_url)
        else:
            self.storage = InMemoryStorage()
        
        # Core processing components
        self.prompt_builder = PromptBuilder()
        self.query_rewriter = QueryRewriter() if self.config.enable_query_rewriting else None
        self.retrieval_filter = RetrievalFilter() if self.config.enable_response_filtering else None
        
        # Background cleanup task
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes
        
        self.logger.info("Context Manager Service initialized")
    
    # Session Management
    def start_session(self, 
                     user_id: str,
                     diagram_id: Optional[str] = None,
                     preferences_override: Dict[str, Any] = None) -> Session:
        """
        Start a new conversation session.
        
        Args:
            user_id: User identifier
            diagram_id: Optional diagram context
            preferences_override: Session-specific preference overrides
            
        Returns:
            New session instance
        """
        try:
            # Generate unique session ID
            session_id = f"session_{uuid.uuid4().hex[:12]}"
            
            # Create session
            session = Session(
                session_id=session_id,
                user_id=user_id,
                diagram_id=diagram_id,
                status=SessionStatus.ACTIVE,
                preferences_override=preferences_override or {}
            )
            
            # Store session
            self.storage.store_session(session)
            
            # Update user activity
            user = self.get_or_create_user(user_id)
            user.total_sessions += 1
            user.update_activity()
            self.storage.update_user(user)
            
            # Record metrics
            self.metrics.counter('sessions_started', tags={'user_id': user_id})
            
            self.logger.info(f"Started new session {session_id} for user {user_id}")
            return session
            
        except Exception as e:
            self.logger.error(f"Failed to start session for user {user_id}: {e}")
            raise ProcessingError(f"Session creation failed: {e}")
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        try:
            session = self.storage.get_session(session_id)
            
            if session and session.should_expire(self.config.session_timeout_minutes):
                # Mark as expired
                session.status = SessionStatus.EXPIRED
                self.storage.update_session(session)
                
                self.logger.info(f"Session {session_id} expired due to inactivity")
                return None
            
            return session
            
        except Exception as e:
            self.logger.error(f"Failed to get session {session_id}: {e}")
            return None
    
    def update_session_activity(self, 
                               session_id: str,
                               query: str = None,
                               response: str = None,
                               entities_mentioned: List[str] = None) -> Optional[Session]:
        """
        Update session with new activity.
        
        Args:
            session_id: Session identifier
            query: User query (if any)
            response: System response (if any)
            entities_mentioned: Entities discussed
            
        Returns:
            Updated session or None if not found
        """
        try:
            session = self.get_session(session_id)
            if not session:
                return None
            
            # Update activity timestamp
            session.update_activity()
            session.message_count += 1 if query else 0
            
            # Update active entities
            if entities_mentioned:
                for entity in entities_mentioned:
                    session.add_entity(entity)
            
            # Store updated session
            self.storage.update_session(session)
            
            # Store messages if provided
            if query:
                message = Message(
                    message_id=f"msg_{uuid.uuid4().hex[:12]}",
                    session_id=session_id,
                    content=query,
                    message_type=MessageType.USER_QUERY,
                    entities_mentioned=entities_mentioned or []
                )
                self.storage.store_message(message)
            
            if response:
                message = Message(
                    message_id=f"msg_{uuid.uuid4().hex[:12]}",
                    session_id=session_id,
                    content=response,
                    message_type=MessageType.SYSTEM_RESPONSE
                )
                self.storage.store_message(message)
            
            return session
            
        except Exception as e:
            self.logger.error(f"Failed to update session activity {session_id}: {e}")
            return session  # Return original session if update fails
    
    def end_session(self, session_id: str) -> bool:
        """
        End a session gracefully.
        
        Args:
            session_id: Session to end
            
        Returns:
            True if successfully ended
        """
        try:
            session = self.get_session(session_id)
            if not session:
                return False
            
            session.status = SessionStatus.INACTIVE
            self.storage.update_session(session)
            
            # Record metrics
            self.metrics.timer('session_duration', session.duration_minutes)
            
            self.logger.info(f"Ended session {session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to end session {session_id}: {e}")
            return False
    
    # User Management
    def get_or_create_user(self, user_id: str) -> User:
        """Get existing user or create new one."""
        try:
            user = self.storage.get_user(user_id)
            
            if not user:
                # Create new user with defaults
                user = User(
                    user_id=user_id,
                    response_style=ResponseStyle.DETAILED,
                    expertise_level=ExpertiseLevel.INTERMEDIATE
                )
                self.storage.store_user(user)
                
                self.logger.info(f"Created new user: {user_id}")
            
            return user
            
        except Exception as e:
            self.logger.error(f"Failed to get/create user {user_id}: {e}")
            raise ProcessingError(f"User management failed: {e}")
    
    def update_user_preferences(self, 
                               user_id: str,
                               preferences: Dict[str, Any]) -> Optional[User]:
        """
        Update user preferences.
        
        Args:
            user_id: User identifier
            preferences: Preferences to update
            
        Returns:
            Updated user or None if not found
        """
        try:
            user = self.storage.get_user(user_id)
            if not user:
                return None
            
            # Update preferences
            user.preferences.update(preferences)
            
            # Update specific preference fields if provided
            if 'response_style' in preferences:
                user.response_style = ResponseStyle(preferences['response_style'])
            
            if 'expertise_level' in preferences:
                user.expertise_level = ExpertiseLevel(preferences['expertise_level'])
            
            user.update_activity()
            self.storage.update_user(user)
            
            self.logger.info(f"Updated preferences for user {user_id}")
            return user
            
        except Exception as e:
            self.logger.error(f"Failed to update user preferences {user_id}: {e}")
            return None
    
    def learn_from_interaction(self,
                             user_id: str,
                             query: str,
                             response_feedback: Dict[str, Any]) -> None:
        """
        Learn implicit preferences from user interactions.
        
        Args:
            user_id: User identifier
            query: User's query
            response_feedback: Feedback about the response
        """
        if not self.config.enable_implicit_learning:
            return
        
        try:
            user = self.storage.get_user(user_id)
            if not user:
                return
            
            # Simple learning heuristics
            if response_feedback.get('too_technical', False):
                # User found response too technical
                current_level = user.expertise_level
                if current_level != ExpertiseLevel.BEGINNER:
                    # Adjust learned preference toward simpler explanations
                    user.learned_preferences['explanation_complexity'] = 'simpler'
            
            elif response_feedback.get('too_simple', False):
                # User found response too simple
                if user.expertise_level != ExpertiseLevel.EXPERT:
                    user.learned_preferences['explanation_complexity'] = 'more_detailed'
            
            if response_feedback.get('too_long', False):
                user.learned_preferences['preferred_length'] = 'shorter'
            elif response_feedback.get('too_short', False):
                user.learned_preferences['preferred_length'] = 'longer'
            
            # Update entity interests
            query_entities = self._extract_entities_from_text(query)
            for entity in query_entities:
                user.update_entity_frequency(entity)
            
            self.storage.update_user(user)
            
            self.logger.debug(f"Learned from interaction for user {user_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to learn from interaction for user {user_id}: {e}")
    
    # Query Processing
    def enhance_query(self,
                     session_id: str,
                     raw_query: str) -> ContextQuery:
        """
        Enhance a user query with contextual information.
        
        Args:
            session_id: Session identifier
            raw_query: Original user query
            
        Returns:
            Enhanced query with context
        """
        try:
            session = self.get_session(session_id)
            if not session:
                # Return unenhanced query if no session
                return ContextQuery(
                    original_query=raw_query,
                    enhanced_query=raw_query,
                    session_id=session_id,
                    enhancement_confidence=0.0
                )
            
            user = self.storage.get_user(session.user_id)
            if not user:
                user = self.get_or_create_user(session.user_id)
            
            # Get conversation history for context
            conversation_history = self.storage.get_session_messages(
                session_id, 
                limit=self.config.max_conversation_history // 10  # Recent messages only
            )
            
            # Enhance query if rewriter is available
            if self.query_rewriter:
                enhanced_query = self.query_rewriter.rewrite_with_context(
                    raw_query, session, user, conversation_history
                )
            else:
                enhanced_query = ContextQuery(
                    original_query=raw_query,
                    enhanced_query=raw_query,
                    session_id=session_id,
                    enhancement_confidence=1.0
                )
            
            # Record metrics
            if enhanced_query.was_enhanced:
                self.metrics.counter('queries_enhanced')
            
            return enhanced_query
            
        except Exception as e:
            self.logger.error(f"Failed to enhance query for session {session_id}: {e}")
            # Return unenhanced query on error
            return ContextQuery(
                original_query=raw_query,
                enhanced_query=raw_query,
                session_id=session_id,
                enhancement_confidence=0.0
            )
    
    def build_context_prompt(self,
                           session_id: str,
                           user_query: str,
                           retrieved_data: List[Dict[str, Any]] = None,
                           prompt_type: str = 'search_query') -> str:
        """
        Build a context-aware prompt for LLM interaction.
        
        Args:
            session_id: Session identifier
            user_query: User's query
            retrieved_data: Retrieved search results
            prompt_type: Type of prompt to build
            
        Returns:
            Context-enriched prompt
        """
        try:
            session = self.get_session(session_id)
            if not session:
                return f"Please help with this query: {user_query}"
            
            user = self.storage.get_user(session.user_id)
            if not user:
                user = self.get_or_create_user(session.user_id)
            
            # Get recent conversation history
            conversation_history = self.storage.get_session_messages(session_id, limit=5)
            
            # Build context-aware prompt
            prompt = self.prompt_builder.build_context_prompt(
                user_query=user_query,
                session=session,
                user=user,
                retrieved_data=retrieved_data,
                conversation_history=conversation_history,
                prompt_type=prompt_type
            )
            
            return prompt
            
        except Exception as e:
            self.logger.error(f"Failed to build context prompt for session {session_id}: {e}")
            return f"Please help with this query: {user_query}"
    
    def filter_results(self,
                      session_id: str,
                      search_results: List[Dict[str, Any]],
                      relevance_threshold: float = None) -> List[Dict[str, Any]]:
        """
        Filter and re-rank search results based on context.
        
        Args:
            session_id: Session identifier
            search_results: Raw search results
            relevance_threshold: Minimum relevance score
            
        Returns:
            Filtered and re-ranked results
        """
        if not self.retrieval_filter or not search_results:
            return search_results
        
        try:
            session = self.get_session(session_id)
            if not session:
                return search_results
            
            user = self.storage.get_user(session.user_id)
            if not user:
                user = self.get_or_create_user(session.user_id)
            
            threshold = relevance_threshold or self.config.relevance_threshold
            
            filtered_results = self.retrieval_filter.filter_by_context(
                search_results, user, session, threshold
            )
            
            # Record metrics
            self.metrics.gauge('results_filtered_ratio', 
                             len(filtered_results) / len(search_results) if search_results else 0)
            
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Failed to filter results for session {session_id}: {e}")
            return search_results
    
    # Conversation Management
    def get_conversation_history(self,
                               session_id: str,
                               limit: int = None) -> List[Message]:
        """Get conversation history for a session."""
        try:
            limit = limit or self.config.max_conversation_history
            return self.storage.get_session_messages(session_id, limit)
            
        except Exception as e:
            self.logger.error(f"Failed to get conversation history for session {session_id}: {e}")
            return []
    
    def summarize_conversation(self, session_id: str) -> Optional[ConversationSummary]:
        """
        Generate a summary of the conversation.
        
        Args:
            session_id: Session to summarize
            
        Returns:
            Conversation summary or None if failed
        """
        try:
            session = self.get_session(session_id)
            if not session:
                return None
            
            messages = self.storage.get_session_messages(session_id)
            if not messages:
                return None
            
            # Build conversation text for summarization
            conversation_text = self._format_conversation_for_summary(messages)
            
            # Generate summary using model client
            summary_prompt = f"""
            Summarize this conversation about network diagrams and technical systems:
            
            {conversation_text}
            
            Provide:
            1. A brief summary of what was discussed
            2. Key topics covered
            3. Important entities mentioned
            
            Keep the summary concise but comprehensive.
            """
            
            summary_text = self.model_client.generate_text(
                prompt=summary_prompt,
                max_tokens=512,
                temperature=0.1
            )
            
            # Extract entities and topics
            entities = list(session.active_entities)
            topics = self._extract_topics_from_messages(messages)
            
            summary = ConversationSummary(
                session_id=session_id,
                user_id=session.user_id,
                summary_text=summary_text,
                key_topics=topics,
                key_entities=entities,
                message_count=len(messages),
                completeness_score=0.8,  # Placeholder
                coherence_score=0.8      # Placeholder
            )
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to summarize conversation for session {session_id}: {e}")
            return None
    
    # Maintenance and Analytics
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        try:
            count = self.storage.cleanup_expired_sessions(self.config.session_timeout_minutes)
            
            self.metrics.gauge('expired_sessions_cleaned', count)
            self._last_cleanup = time.time()
            
            return count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        try:
            storage_stats = self.storage.get_stats()
            
            # Check if cleanup is needed
            should_cleanup = time.time() - self._last_cleanup > self._cleanup_interval
            if should_cleanup and self.config.auto_cleanup_enabled:
                self.cleanup_expired_sessions()
            
            return {
                'service': 'context_manager',
                'config': {
                    'session_timeout_minutes': self.config.session_timeout_minutes,
                    'max_conversation_history': self.config.max_conversation_history,
                    'enable_implicit_learning': self.config.enable_implicit_learning,
                    'enable_query_rewriting': self.config.enable_query_rewriting,
                },
                'storage': storage_stats,
                'last_cleanup': self._last_cleanup,
                'should_cleanup': should_cleanup
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get service statistics: {e}")
            return {'error': str(e)}
    
    # Helper methods
    def _extract_entities_from_text(self, text: str) -> List[str]:
        """Simple entity extraction from text."""
        # This is a placeholder - in production, you'd use NER or similar
        technical_terms = [
            'server', 'database', 'router', 'switch', 'firewall', 
            'load balancer', 'network', 'protocol', 'api', 'service'
        ]
        
        text_lower = text.lower()
        found_entities = [term for term in technical_terms if term in text_lower]
        
        return found_entities
    
    def _format_conversation_for_summary(self, messages: List[Message]) -> str:
        """Format conversation messages for summarization."""
        formatted_lines = []
        
        for message in messages[-20:]:  # Last 20 messages
            role = "User" if message.message_type == MessageType.USER_QUERY else "System"
            content = message.content[:200] + "..." if len(message.content) > 200 else message.content
            formatted_lines.append(f"{role}: {content}")
        
        return "\\n".join(formatted_lines)
    
    def _extract_topics_from_messages(self, messages: List[Message]) -> List[str]:
        """Extract key topics from conversation messages."""
        # Simple topic extraction - could be enhanced with ML
        topic_keywords = {
            'networking': ['network', 'router', 'switch', 'protocol', 'connection'],
            'security': ['firewall', 'security', 'authentication', 'encryption'],
            'infrastructure': ['server', 'database', 'architecture', 'topology'],
            'monitoring': ['monitoring', 'logging', 'metrics', 'alerts']
        }
        
        all_text = " ".join(msg.content.lower() for msg in messages)
        
        topics = []
        for topic, keywords in topic_keywords.items():
            if any(keyword in all_text for keyword in keywords):
                topics.append(topic)
        
        return topics