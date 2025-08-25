"""
Context Manager integration for TextRAG.

Provides enhanced text search capabilities by integrating with the
Context Manager service for conversation-aware retrieval.
"""

from typing import List, Optional, Dict, Any
import asyncio

from ....shared import get_logger
from ...context_manager import ContextManager
from ..client import TextRAG
from ..models import SearchResultSet, SearchQuery, SearchMethod


class ContextAwareTextRAG:
    """
    Context-aware TextRAG integration.
    
    Enhances TextRAG with conversation context and user preference
    awareness through integration with the Context Manager.
    """
    
    def __init__(self, 
                 textrag: Optional[TextRAG] = None,
                 context_manager: Optional[ContextManager] = None):
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.textrag = textrag or TextRAG()
        self.context_manager = context_manager or ContextManager()
        
        self.logger.info("Context-aware TextRAG initialized")
    
    async def conversational_search(self, 
                                   query: str,
                                   session_id: str,
                                   top_k: int = 5,
                                   similarity_threshold: float = 0.7) -> SearchResultSet:
        """
        Perform conversation-aware text search.
        
        Args:
            query: Search query
            session_id: Context Manager session ID
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            Search results enhanced with conversation context
        """
        try:
            # Get session information from Context Manager
            session = self.context_manager.get_session(session_id)
            
            if not session:
                self.logger.warning(f"Session not found: {session_id}")
                # Fall back to regular search
                return await self.textrag.search(query, top_k, similarity_threshold)
            
            # Get conversation history for context
            conversation_history = self.context_manager.get_conversation_history(session_id, limit=5)
            context_messages = [msg.content for msg in conversation_history if msg.content]
            
            # Enhance query using Context Manager
            enhanced_query = self.context_manager.enhance_query(session_id, query)
            
            # Perform context-aware search
            results = await self.textrag.conversational_search(
                query=enhanced_query.enhanced_query,
                session_id=session_id,
                context_messages=context_messages,
                top_k=top_k
            )
            
            # Update session with search activity
            retrieved_context = [
                {'chunk_id': result.chunk_id, 'document_title': result.document_title}
                for result in results.results
            ]
            
            self.context_manager.update_session(
                session_id=session_id,
                query=query,
                response=f"Found {len(results.results)} relevant documents",
                retrieved_context=retrieved_context
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Conversational search failed: {e}")
            # Fall back to regular search
            return await self.textrag.search(query, top_k, similarity_threshold)
    
    async def user_personalized_search(self, 
                                      query: str,
                                      user_id: str,
                                      session_id: Optional[str] = None,
                                      top_k: int = 5) -> SearchResultSet:
        """
        Perform user-personalized text search.
        
        Args:
            query: Search query
            user_id: User ID for personalization
            session_id: Optional session ID for context
            top_k: Number of results to return
            
        Returns:
            Personalized search results
        """
        try:
            # Get user preferences from Context Manager
            user = self.context_manager.get_user(user_id)
            
            # Adjust search parameters based on user preferences
            adjusted_top_k = top_k
            adjusted_threshold = 0.7
            
            if user:
                # Adjust based on user expertise level
                expertise = user.get_preference('expertise_level', 'intermediate')
                if expertise == 'beginner':
                    adjusted_threshold = 0.6  # More lenient for beginners
                elif expertise == 'expert':
                    adjusted_threshold = 0.8  # More strict for experts
                
                # Adjust result count based on user preferences
                response_style = user.get_preference('response_style', 'detailed')
                if response_style == 'brief':
                    adjusted_top_k = min(top_k, 3)
                elif response_style == 'detailed':
                    adjusted_top_k = max(top_k, 7)
            
            # Perform search with personalized parameters
            if session_id:
                results = await self.conversational_search(
                    query, session_id, adjusted_top_k, adjusted_threshold
                )
            else:
                results = await self.textrag.search(
                    query, adjusted_top_k, similarity_threshold=adjusted_threshold
                )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Personalized search failed: {e}")
            return await self.textrag.search(query, top_k)
    
    async def multi_turn_search(self, 
                               queries: List[str],
                               session_id: str,
                               combine_results: bool = True) -> List[SearchResultSet]:
        """
        Perform multiple related searches in sequence.
        
        Args:
            queries: List of search queries
            session_id: Context Manager session ID
            combine_results: Whether to combine and deduplicate results
            
        Returns:
            List of search result sets
        """
        try:
            results = []
            
            for query in queries:
                # Each search builds on the previous conversation context
                result_set = await self.conversational_search(query, session_id)
                results.append(result_set)
                
                # Small delay to allow context to be processed
                await asyncio.sleep(0.1)
            
            if combine_results and len(results) > 1:
                # Combine and deduplicate results
                combined_results = self._combine_search_results(results)
                return [combined_results]
            
            return results
            
        except Exception as e:
            self.logger.error(f"Multi-turn search failed: {e}")
            return []
    
    async def search_with_feedback(self, 
                                  query: str,
                                  session_id: str,
                                  feedback: Optional[Dict[str, Any]] = None,
                                  top_k: int = 5) -> SearchResultSet:
        """
        Perform search with user feedback integration.
        
        Args:
            query: Search query
            session_id: Context Manager session ID
            feedback: User feedback on previous results
            top_k: Number of results to return
            
        Returns:
            Search results adjusted for feedback
        """
        try:
            # Apply feedback to search parameters if provided
            adjusted_threshold = 0.7
            adjusted_query = query
            
            if feedback:
                # Adjust search based on feedback
                if feedback.get('relevance') == 'low':
                    adjusted_threshold = 0.6  # More lenient
                elif feedback.get('relevance') == 'high':
                    adjusted_threshold = 0.8  # More strict
                
                # Use feedback to enhance query
                if feedback.get('preferred_topics'):
                    topics = ' '.join(feedback['preferred_topics'])
                    adjusted_query = f"{query} {topics}"
            
            # Perform search
            results = await self.conversational_search(
                adjusted_query, session_id, top_k, adjusted_threshold
            )
            
            # Record feedback for learning
            if feedback:
                self.context_manager.update_session(
                    session_id=session_id,
                    user_feedback=feedback
                )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Feedback-aware search failed: {e}")
            return await self.conversational_search(query, session_id, top_k)
    
    def get_search_insights(self, session_id: str) -> Dict[str, Any]:
        """
        Get insights about user's search patterns and preferences.
        
        Args:
            session_id: Context Manager session ID
            
        Returns:
            Dictionary with search insights
        """
        try:
            # Get session analytics
            session_analytics = self.context_manager.get_session_analytics(session_id)
            
            # Get user insights
            session = self.context_manager.get_session(session_id)
            if session:
                user_insights = self.context_manager.get_user_insights(session.user_id)
            else:
                user_insights = {}
            
            return {
                'session_analytics': session_analytics,
                'user_insights': user_insights,
                'search_patterns': self._analyze_search_patterns(session_id)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get search insights: {e}")
            return {}
    
    def _combine_search_results(self, result_sets: List[SearchResultSet]) -> SearchResultSet:
        """Combine multiple search result sets, removing duplicates."""
        if not result_sets:
            return SearchResultSet(
                query_id="combined",
                query_text="combined_search",
                search_method=SearchMethod.CONTEXT_AWARE,
                results=[],
                total_results=0
            )
        
        # Combine all results
        all_results = []
        seen_chunk_ids = set()
        
        for result_set in result_sets:
            for result in result_set.results:
                if result.chunk_id not in seen_chunk_ids:
                    all_results.append(result)
                    seen_chunk_ids.add(result.chunk_id)
        
        # Sort by fusion score
        all_results.sort(key=lambda r: r.fusion_score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(all_results):
            result.final_rank = i + 1
        
        # Create combined result set
        combined_query = " | ".join([rs.query_text for rs in result_sets])
        total_time = sum([rs.search_time_ms for rs in result_sets])
        
        return SearchResultSet(
            query_id="combined",
            query_text=combined_query,
            search_method=SearchMethod.CONTEXT_AWARE,
            results=all_results,
            total_results=len(all_results),
            search_time_ms=total_time
        )
    
    def _analyze_search_patterns(self, session_id: str) -> Dict[str, Any]:
        """Analyze search patterns for the session."""
        try:
            # Get conversation history
            history = self.context_manager.get_conversation_history(session_id)
            
            # Analyze query patterns
            queries = [msg.content for msg in history if 'search' in msg.content.lower()]
            
            patterns = {
                'total_queries': len(queries),
                'common_terms': self._extract_common_terms(queries),
                'query_complexity': self._analyze_query_complexity(queries),
                'topic_trends': self._identify_topic_trends(queries)
            }
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to analyze search patterns: {e}")
            return {}
    
    def _extract_common_terms(self, queries: List[str]) -> List[str]:
        """Extract common terms from queries."""
        if not queries:
            return []
        
        # Simple term frequency analysis
        term_freq = {}
        for query in queries:
            words = query.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    term_freq[word] = term_freq.get(word, 0) + 1
        
        # Return top terms
        sorted_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)
        return [term for term, freq in sorted_terms[:10]]
    
    def _analyze_query_complexity(self, queries: List[str]) -> Dict[str, float]:
        """Analyze the complexity of queries."""
        if not queries:
            return {}
        
        total_words = sum(len(q.split()) for q in queries)
        avg_words = total_words / len(queries)
        
        long_queries = sum(1 for q in queries if len(q.split()) > 10)
        complexity_ratio = long_queries / len(queries)
        
        return {
            'average_words_per_query': avg_words,
            'complex_query_ratio': complexity_ratio
        }
    
    def _identify_topic_trends(self, queries: List[str]) -> List[str]:
        """Identify trending topics in queries."""
        # This would use NLP to identify topics
        # For now, return a simple keyword-based analysis
        
        technical_terms = [
            'firewall', 'router', 'switch', 'network', 'server', 'database',
            'security', 'configuration', 'protocol', 'connection'
        ]
        
        topic_counts = {}
        for query in queries:
            query_lower = query.lower()
            for term in technical_terms:
                if term in query_lower:
                    topic_counts[term] = topic_counts.get(term, 0) + 1
        
        # Return trending topics
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, count in sorted_topics[:5]]
    
    def close(self) -> None:
        """Clean up resources."""
        try:
            self.textrag.close()
            self.context_manager.close()
            self.logger.info("Context-aware TextRAG closed")
        except Exception as e:
            self.logger.error(f"Error closing context-aware TextRAG: {e}")