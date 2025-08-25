"""
Context Manager client interface.

Provides a simple, clean interface for integrating context management
into RAG systems and other conversational AI applications.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from ...shared import get_logger
from .service import ContextManagerService
from .models import (
    Session, Message, User, ContextQuery, ContextConfig,
    ResponseStyle, ExpertiseLevel
)
from .storage.backends import StorageBackend, InMemoryStorage, ExternalStorage


class ContextManager:
    """
    Main client interface for Context Manager functionality.
    
    Provides a simplified API for:
    - Session management
    - Query enhancement
    - Context-aware processing
    - User preference management
    """
    
    def __init__(self,
                 config: ContextConfig = None,
                 storage_backend: StorageBackend = None,
                 enable_learning: bool = True):
        """
        Initialize Context Manager client.
        
        Args:
            config: Configuration settings
            storage_backend: Storage backend to use
            enable_learning: Enable implicit preference learning
        """
        self.logger = get_logger(__name__)
        
        # Set up configuration
        if config is None:
            config = ContextConfig()
        config.enable_implicit_learning = enable_learning
        
        # Initialize the service
        self.service = ContextManagerService(
            config=config,
            storage_backend=storage_backend
        )
        
        self.logger.info("Context Manager client initialized")
    
    # Session Management API
    def start_session(self,
                     user_id: str,
                     diagram_id: Optional[str] = None,
                     user_preferences: Dict[str, Any] = None) -> Session:
        """
        Start a new conversation session.
        
        Args:
            user_id: User identifier
            diagram_id: Optional diagram context
            user_preferences: Session-specific preferences
            
        Returns:
            New session object
            
        Example:
            >>> context_manager = ContextManager()
            >>> session = context_manager.start_session(
            ...     user_id="user_123",
            ...     diagram_id="network_001"
            ... )
            >>> print(session.session_id)
            session_abc123
        """
        try:
            return self.service.start_session(
                user_id=user_id,
                diagram_id=diagram_id,
                preferences_override=user_preferences
            )
        except Exception as e:
            self.logger.error(f"Failed to start session: {e}")
            raise
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session object or None if not found/expired
        """
        return self.service.get_session(session_id)
    
    def end_session(self, session_id: str) -> bool:
        """
        End a session gracefully.
        
        Args:
            session_id: Session to end
            
        Returns:
            True if successfully ended
        """
        return self.service.end_session(session_id)
    
    def get_active_sessions(self, user_id: str) -> List[Session]:
        """
        Get all active sessions for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of active sessions
        """
        try:
            return self.service.storage.get_active_sessions(user_id)
        except Exception as e:
            self.logger.error(f"Failed to get active sessions: {e}")
            return []
    
    # Query Enhancement API  
    def enhance_query(self,
                     session_id: str,
                     raw_query: str) -> ContextQuery:
        """
        Enhance a user query with contextual information.
        
        Args:
            session_id: Session identifier
            raw_query: Original user query
            
        Returns:
            Enhanced query with context metadata
            
        Example:
            >>> enhanced = context_manager.enhance_query(
            ...     session_id="session_123",
            ...     raw_query="Show me that firewall"
            ... )
            >>> print(enhanced.enhanced_query)
            Show me the firewall in diagram network_001
            >>> print(enhanced.enhancements_applied)
            ['pronoun_resolution', 'diagram_context']
        """
        return self.service.enhance_query(session_id, raw_query)
    
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
            prompt_type: Type of prompt ('search_query', 'explanation', etc.)
            
        Returns:
            Context-enriched prompt
            
        Example:
            >>> prompt = context_manager.build_context_prompt(
            ...     session_id="session_123",
            ...     user_query="Explain this network topology",
            ...     retrieved_data=search_results
            ... )
        """
        return self.service.build_context_prompt(
            session_id, user_query, retrieved_data, prompt_type
        )
    
    def filter_results(self,
                      session_id: str,
                      search_results: List[Dict[str, Any]],
                      relevance_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        Filter and re-rank search results based on context.
        
        Args:
            session_id: Session identifier
            search_results: Raw search results
            relevance_threshold: Minimum relevance score to keep
            
        Returns:
            Filtered and contextually re-ranked results
            
        Example:
            >>> filtered = context_manager.filter_results(
            ...     session_id="session_123",
            ...     search_results=raw_results,
            ...     relevance_threshold=0.7
            ... )
        """
        return self.service.filter_results(session_id, search_results, relevance_threshold)
    
    # Session Update API
    def update_session(self,
                      session_id: str,
                      query: str = None,
                      response: str = None,
                      retrieved_context: List[Dict[str, Any]] = None,
                      user_feedback: Dict[str, Any] = None) -> Optional[Session]:
        """
        Update session with conversation activity.
        
        Args:
            session_id: Session identifier
            query: User query (if any)
            response: System response (if any)  
            retrieved_context: Retrieved data/context
            user_feedback: User feedback for learning
            
        Returns:
            Updated session or None if not found
            
        Example:
            >>> session = context_manager.update_session(
            ...     session_id="session_123",
            ...     query="Find load balancers",
            ...     response="Found 3 load balancers in the network",
            ...     retrieved_context=search_results
            ... )
        """
        try:
            # Extract entities from retrieved context
            entities_mentioned = []
            if retrieved_context:
                for item in retrieved_context:
                    if isinstance(item, dict):
                        if 'label' in item:
                            entities_mentioned.append(item['label'])
                        elif 'name' in item:
                            entities_mentioned.append(item['name'])
            
            # Update session activity
            session = self.service.update_session_activity(
                session_id=session_id,
                query=query,
                response=response,
                entities_mentioned=entities_mentioned
            )
            
            # Learn from user feedback if provided
            if user_feedback and session:
                self.service.learn_from_interaction(
                    session.user_id, query or "", user_feedback
                )
            
            return session
            
        except Exception as e:
            self.logger.error(f"Failed to update session: {e}")
            return None
    
    # User Management API
    def get_user(self, user_id: str) -> User:
        """
        Get or create user profile.
        
        Args:
            user_id: User identifier
            
        Returns:
            User profile
        """
        return self.service.get_or_create_user(user_id)
    
    def update_user_preferences(self,
                               user_id: str,
                               preferences: Dict[str, Any]) -> Optional[User]:
        """
        Update user preferences.
        
        Args:
            user_id: User identifier
            preferences: Preferences to update
            
        Returns:
            Updated user profile
            
        Example:
            >>> user = context_manager.update_user_preferences(
            ...     user_id="user_123",
            ...     preferences={
            ...         "response_style": "technical",
            ...         "expertise_level": "advanced"
            ...     }
            ... )
        """
        return self.service.update_user_preferences(user_id, preferences)
    
    def set_user_expertise(self, user_id: str, expertise_level: str) -> Optional[User]:
        """
        Set user expertise level.
        
        Args:
            user_id: User identifier
            expertise_level: "beginner", "intermediate", "advanced", or "expert"
            
        Returns:
            Updated user profile
        """
        try:
            expertise = ExpertiseLevel(expertise_level.lower())
            return self.update_user_preferences(
                user_id, 
                {"expertise_level": expertise.value}
            )
        except ValueError:
            self.logger.error(f"Invalid expertise level: {expertise_level}")
            return None
    
    def set_user_response_style(self, user_id: str, response_style: str) -> Optional[User]:
        """
        Set user response style preference.
        
        Args:
            user_id: User identifier
            response_style: "brief", "detailed", "technical", or "conversational"
            
        Returns:
            Updated user profile
        """
        try:
            style = ResponseStyle(response_style.lower())
            return self.update_user_preferences(
                user_id,
                {"response_style": style.value}
            )
        except ValueError:
            self.logger.error(f"Invalid response style: {response_style}")
            return None
    
    # Conversation History API
    def get_conversation_history(self,
                               session_id: str,
                               limit: int = 50) -> List[Message]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return
            
        Returns:
            List of messages in chronological order
        """
        return self.service.get_conversation_history(session_id, limit)
    
    def summarize_conversation(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Generate a summary of the conversation.
        
        Args:
            session_id: Session to summarize
            
        Returns:
            Conversation summary dictionary
        """
        try:
            summary = self.service.summarize_conversation(session_id)
            return summary.dict() if summary else None
        except Exception as e:
            self.logger.error(f"Failed to summarize conversation: {e}")
            return None
    
    # Analytics and Maintenance API
    def get_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """
        Get analytics for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session analytics data
        """
        try:
            return self.service.storage.repository.get_session_statistics(session_id)
        except Exception as e:
            self.logger.error(f"Failed to get session analytics: {e}")
            return {}
    
    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """
        Get behavioral insights for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            User insights and patterns
        """
        try:
            return self.service.storage.repository.get_user_insights(user_id)
        except Exception as e:
            self.logger.error(f"Failed to get user insights: {e}")
            return {}
    
    def get_service_stats(self) -> Dict[str, Any]:
        """
        Get overall service statistics.
        
        Returns:
            Service statistics and health info
        """
        return self.service.get_service_statistics()
    
    def cleanup_expired_sessions(self) -> int:
        """
        Manually trigger cleanup of expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        return self.service.cleanup_expired_sessions()
    
    # Convenience Methods for Common Workflows
    def process_user_query(self,
                          session_id: str,
                          user_query: str,
                          search_results: List[Dict[str, Any]] = None,
                          user_feedback: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Complete workflow for processing a user query with context.
        
        Args:
            session_id: Session identifier
            user_query: User's query
            search_results: Search results to filter
            user_feedback: User feedback for learning
            
        Returns:
            Complete processing result
            
        Example:
            >>> result = context_manager.process_user_query(
            ...     session_id="session_123",
            ...     user_query="Show me those routers again",
            ...     search_results=raw_search_results
            ... )
            >>> enhanced_query = result['enhanced_query']
            >>> context_prompt = result['context_prompt']
            >>> filtered_results = result['filtered_results']
        """
        try:
            # Step 1: Enhance the query
            enhanced_query = self.enhance_query(session_id, user_query)
            
            # Step 2: Build context prompt
            context_prompt = self.build_context_prompt(
                session_id, 
                enhanced_query.enhanced_query,
                search_results
            )
            
            # Step 3: Filter results if provided
            filtered_results = None
            if search_results:
                filtered_results = self.filter_results(session_id, search_results)
            
            # Step 4: Update session
            session = self.update_session(
                session_id=session_id,
                query=user_query,
                retrieved_context=filtered_results or search_results,
                user_feedback=user_feedback
            )
            
            return {
                'enhanced_query': enhanced_query,
                'context_prompt': context_prompt,
                'filtered_results': filtered_results,
                'session': session,
                'processing_success': True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process user query: {e}")
            return {
                'enhanced_query': None,
                'context_prompt': user_query,
                'filtered_results': search_results,
                'session': None,
                'processing_success': False,
                'error': str(e)
            }
    
    def close(self):
        """Clean up resources."""
        try:
            # Perform final cleanup
            self.cleanup_expired_sessions()
            self.logger.info("Context Manager client closed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    # Context Manager Configuration
    @classmethod
    def create_with_memory_storage(cls, **kwargs) -> 'ContextManager':
        """Create Context Manager with in-memory storage."""
        return cls(storage_backend=InMemoryStorage(), **kwargs)
    
    @classmethod  
    def create_with_database_storage(cls, database_url: str = None, **kwargs) -> 'ContextManager':
        """Create Context Manager with database storage."""
        return cls(storage_backend=ExternalStorage(database_url), **kwargs)