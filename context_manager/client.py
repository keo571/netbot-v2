"""
Main Context Manager class for stateful RAG chatbot systems.

The ContextManager orchestrates all three context stores (session, history, preferences)
and provides the core functionality for query rewriting, context injection, and 
state management as described in docs/architecture/context-manager.md.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict

try:
    # Try relative imports first (for package usage)
    from .models import (
        SessionState, ConversationHistory, UserPreferences, ConversationExchange,
        QueryIntent, ResponseStyle, ExpertiseLevel, UserFeedback
    )
    from .storage import SessionStore, HistoryStore, UserStore
    from .core import QueryRewriter, RetrievalFilter, PromptBuilder
except ImportError:
    # Fall back to absolute imports (for direct script execution)
    from models import (
        SessionState, ConversationHistory, UserPreferences, ConversationExchange,
        QueryIntent, ResponseStyle, ExpertiseLevel, UserFeedback
    )
    from storage import SessionStore, HistoryStore, UserStore
    from core import QueryRewriter, RetrievalFilter, PromptBuilder


logger = logging.getLogger(__name__)


class ContextManager:
    """
    The brain of a stateful RAG chatbot system.
    
    Manages session state, conversation history, and user preferences to enable
    coherent, personalized conversations with memory and context awareness.
    """
    
    def __init__(self, 
                 session_store: SessionStore,
                 history_store: HistoryStore,
                 user_store: UserStore,
                 session_timeout_seconds: int = 1800):
        """
        Initialize the Context Manager with storage backends.
        
        Args:
            session_store: Storage for ephemeral session state
            history_store: Storage for persistent conversation history  
            user_store: Storage for persistent user preferences
            session_timeout_seconds: TTL for session state (default 30 min)
        """
        self.session_store = session_store
        self.history_store = history_store
        self.user_store = user_store
        self.session_timeout = session_timeout_seconds
        
        # Initialize helper components
        self.query_rewriter = QueryRewriter()
        self.retrieval_filter = RetrievalFilter()
        self.prompt_builder = PromptBuilder()
        
        logger.info("ContextManager initialized successfully")
    
    # Session Management
    
    def create_session(self, user_id: str, session_id: Optional[str] = None) -> str:
        """
        Create a new conversation session.
        
        Args:
            user_id: User identifier
            session_id: Optional session ID, will generate if not provided
            
        Returns:
            The session ID
        """
        if not session_id:
            session_id = f"{user_id}_{uuid.uuid4().hex[:8]}"
        
        session = SessionState(session_id=session_id)
        
        if self.session_store.save_session(session, self.session_timeout):
            logger.info(f"Created new session: {session_id}")
            return session_id
        else:
            logger.error(f"Failed to create session: {session_id}")
            raise RuntimeError(f"Failed to create session: {session_id}")
    
    def get_or_create_session(self, user_id: str, session_id: Optional[str] = None) -> SessionState:
        """
        Get existing session or create a new one.
        
        Args:
            user_id: User identifier
            session_id: Optional session ID
            
        Returns:
            SessionState object
        """
        if session_id and self.session_store.exists(session_id):
            session = self.session_store.get_session(session_id)
            if session:
                # Extend session TTL on access
                self.session_store.save_session(session, self.session_timeout)
                return session
        
        # Create new session
        new_session_id = self.create_session(user_id, session_id)
        return self.session_store.get_session(new_session_id)
    
    def end_session(self, session_id: str) -> bool:
        """
        Explicitly end a session.
        
        Args:
            session_id: Session to end
            
        Returns:
            True if session was deleted, False otherwise
        """
        if self.session_store.delete_session(session_id):
            logger.info(f"Ended session: {session_id}")
            return True
        return False
    
    # User Preferences Management
    
    def get_user_preferences(self, user_id: str) -> UserPreferences:
        """
        Get user preferences, creating defaults if none exist.
        
        Args:
            user_id: User identifier
            
        Returns:
            UserPreferences object
        """
        preferences = self.user_store.get_preferences(user_id)
        if not preferences:
            # Create default preferences
            preferences = UserPreferences(user_id=user_id)
            self.user_store.save_preferences(preferences)
            logger.info(f"Created default preferences for user: {user_id}")
        
        return preferences
    
    def update_user_preferences(self, user_id: str, **updates) -> bool:
        """
        Update user preferences.
        
        Args:
            user_id: User identifier
            **updates: Preference fields to update
            
        Returns:
            True if successful, False otherwise
        """
        preferences = self.get_user_preferences(user_id)
        
        # Update provided fields
        for key, value in updates.items():
            if hasattr(preferences, key):
                setattr(preferences, key, value)
        
        preferences.updated_at = datetime.now()
        
        if self.user_store.save_preferences(preferences):
            logger.info(f"Updated preferences for user: {user_id}")
            return True
        return False
    
    # Conversation History Management
    
    def get_conversation_history(self, user_id: str) -> ConversationHistory:
        """
        Get conversation history, creating empty if none exists.
        
        Args:
            user_id: User identifier
            
        Returns:
            ConversationHistory object
        """
        history = self.history_store.get_history(user_id)
        if not history:
            history = ConversationHistory(user_id=user_id)
            logger.info(f"Created new conversation history for user: {user_id}")
        
        return history
    
    def add_conversation_exchange(self, 
                                user_id: str,
                                query: str,
                                response: str,
                                retrieved_context: Dict[str, Any],
                                diagram_ids_used: List[str] = None,
                                user_feedback: UserFeedback = UserFeedback.NONE) -> bool:
        """
        Add a new exchange to conversation history.
        
        Args:
            user_id: User identifier
            query: User's query
            response: LLM response
            retrieved_context: Context used for response
            diagram_ids_used: Diagram IDs referenced
            user_feedback: User feedback on response
            
        Returns:
            True if successful, False otherwise
        """
        history = self.get_conversation_history(user_id)
        
        exchange = ConversationExchange(
            query_text=query,
            llm_response=response,
            retrieved_context=retrieved_context,
            timestamp=datetime.now(),
            user_feedback=user_feedback,
            diagram_ids_used=diagram_ids_used or []
        )
        
        history.add_exchange(exchange)
        
        if self.history_store.save_history(history):
            logger.info(f"Added conversation exchange for user: {user_id}")
            return True
        return False
    
    # Core Context Processing Methods
    
    def process_query(self, 
                     user_id: str,
                     raw_query: str,
                     session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user query through the complete context pipeline.
        
        This is the main method that orchestrates the entire context-aware
        query processing workflow.
        
        Args:
            user_id: User identifier
            raw_query: User's raw query text
            session_id: Optional session ID
            
        Returns:
            Dict containing processed query info and context
        """
        # Get or create session
        session = self.get_or_create_session(user_id, session_id)
        
        # Get user preferences and history
        preferences = self.get_user_preferences(user_id)
        history = self.get_conversation_history(user_id)
        
        # Step 1: Query rewriting and expansion
        rewritten_query = self.query_rewriter.rewrite_query(
            raw_query, session, history, preferences
        )
        
        # Step 2: Classify query intent
        query_intent = self.query_rewriter.classify_intent(raw_query, session)
        
        # Update session with query intent
        session.last_query_intent = query_intent
        session.timestamp = datetime.now()
        
        # Step 3: Extract entities from query
        entities = self.query_rewriter.extract_entities(rewritten_query)
        
        # Step 4: Analyze expertise signals
        expertise_signals = self.query_rewriter.analyze_expertise_signals(raw_query, session)
        
        # Update user preferences with expertise signals
        preferences.update_expertise_signals(**expertise_signals)
        self.user_store.save_preferences(preferences)
        
        return {
            'session_id': session.session_id,
            'original_query': raw_query,
            'rewritten_query': rewritten_query,
            'query_intent': query_intent,
            'extracted_entities': entities,
            'session_state': session,
            'user_preferences': preferences,
            'recent_history': history.get_recent_exchanges(3),
            'context_ready': True
        }
    
    def filter_and_rank_results(self,
                              retrieval_results: List[Dict[str, Any]],
                              user_id: str,
                              session_id: str,
                              query_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Filter and re-rank retrieval results based on context.
        
        Args:
            retrieval_results: Raw results from vector search
            user_id: User identifier
            session_id: Session identifier
            query_context: Context from process_query
            
        Returns:
            Filtered and re-ranked results
        """
        preferences = query_context.get('user_preferences')
        session = query_context.get('session_state')
        
        # Apply context-aware filtering and ranking
        filtered_results = self.retrieval_filter.filter_by_preferences(
            retrieval_results, preferences
        )
        
        ranked_results = self.retrieval_filter.rank_by_session_context(
            filtered_results, session
        )
        
        return ranked_results
    
    def build_context_aware_prompt(self,
                                  query_context: Dict[str, Any],
                                  retrieval_results: List[Dict[str, Any]],
                                  graph_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Build a context-aware prompt for the LLM.
        
        Args:
            query_context: Context from process_query
            retrieval_results: Filtered retrieval results
            graph_data: Optional graph/diagram data
            
        Returns:
            Formatted prompt string
        """
        return self.prompt_builder.build_prompt(
            query_context=query_context,
            retrieval_results=retrieval_results,
            graph_data=graph_data
        )
    
    def update_context_after_response(self,
                                    user_id: str,
                                    session_id: str,
                                    query_context: Dict[str, Any],
                                    llm_response: str,
                                    retrieval_context: Dict[str, Any],
                                    diagram_ids_used: List[str] = None,
                                    user_feedback: UserFeedback = UserFeedback.NONE) -> bool:
        """
        Update all context stores after generating a response.
        
        This is critical for maintaining state for future interactions.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            query_context: Context from process_query
            llm_response: Generated response
            retrieval_context: Context used for response
            diagram_ids_used: Diagram IDs referenced
            user_feedback: User feedback
            
        Returns:
            True if all updates successful
        """
        success = True
        
        try:
            # Update session state
            session = query_context['session_state']
            entities = query_context.get('extracted_entities', [])
            
            # Update active entities
            session.update_entities(entities)
            
            # Update diagram IDs if any were used
            if diagram_ids_used:
                session.update_diagram_ids(diagram_ids_used)
            
            # Save updated session
            if not self.session_store.save_session(session, self.session_timeout):
                logger.error(f"Failed to update session: {session_id}")
                success = False
            
            # Add to conversation history
            if not self.add_conversation_exchange(
                user_id=user_id,
                query=query_context['original_query'],
                response=llm_response,
                retrieved_context=retrieval_context,
                diagram_ids_used=diagram_ids_used,
                user_feedback=user_feedback
            ):
                logger.error(f"Failed to update history for user: {user_id}")
                success = False
            
            # Update user preferences implicitly based on feedback
            if user_feedback != UserFeedback.NONE:
                self._update_preferences_from_feedback(user_id, user_feedback, query_context)
            
            logger.info(f"Updated context after response for user: {user_id}")
            
        except Exception as e:
            logger.error(f"Error updating context: {e}")
            success = False
        
        return success
    
    def _update_preferences_from_feedback(self,
                                        user_id: str,
                                        feedback: UserFeedback,
                                        query_context: Dict[str, Any]):
        """
        Update user preferences based on feedback signals.
        
        Args:
            user_id: User identifier
            feedback: User feedback
            query_context: Query context
        """
        try:
            preferences = self.get_user_preferences(user_id)
            
            # Simple feedback-based adjustment
            feedback_signal = 1.0 if feedback == UserFeedback.THUMB_UP else -1.0
            
            # Adjust response style based on feedback
            preferences.adjust_response_style_score(feedback_signal * 0.3)
            
            # Update topic interests based on extracted entities
            entities = query_context.get('extracted_entities', [])
            if entities:
                preferences.update_topic_interests(entities)
            
            self.user_store.save_preferences(preferences)
            
        except Exception as e:
            logger.error(f"Error updating preferences from feedback: {e}")
    
    # Utility Methods
    
    def get_context_summary(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """
        Get a summary of current context state.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            Context summary dictionary
        """
        session = self.session_store.get_session(session_id)
        preferences = self.get_user_preferences(user_id)
        history = self.get_conversation_history(user_id)
        
        return {
            'session_active': session is not None,
            'session_state': asdict(session) if session else None,
            'user_preferences': asdict(preferences),
            'conversation_length': len(history.conversation_log),
            'recent_exchanges': len(history.get_recent_exchanges(5)),
            'last_activity': session.timestamp if session else None
        }
    
    def clear_user_data(self, user_id: str, preserve_preferences: bool = True) -> bool:
        """
        Clear user data (for privacy/GDPR compliance).
        
        Args:
            user_id: User identifier
            preserve_preferences: Whether to keep user preferences
            
        Returns:
            True if successful
        """
        success = True
        
        # Clear conversation history
        if not self.history_store.delete_history(user_id):
            logger.warning(f"Failed to delete history for user: {user_id}")
            success = False
        
        # Clear preferences if requested
        if not preserve_preferences:
            if not self.user_store.delete_preferences(user_id):
                logger.warning(f"Failed to delete preferences for user: {user_id}")
                success = False
        
        logger.info(f"Cleared data for user: {user_id} (preserve_preferences={preserve_preferences})")
        return success