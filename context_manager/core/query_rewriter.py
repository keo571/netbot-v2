"""
Query rewriting and expansion for context-aware retrieval.

Transforms raw user queries into contextually rich versions using session state,
conversation history, and user preferences. Implements the query rewriting
strategies outlined in docs/architecture/context-manager.md.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta

try:
    from ..models import SessionState, ConversationHistory, UserPreferences, QueryIntent
except ImportError:
    from models import SessionState, ConversationHistory, UserPreferences, QueryIntent


logger = logging.getLogger(__name__)


class QueryRewriter:
    """
    Handles query rewriting, entity extraction, and intent classification.
    """
    
    def __init__(self):
        """Initialize the query rewriter with patterns and keywords."""
        
        # Pronouns and demonstratives that indicate context dependency
        self.context_indicators = {
            'pronouns': ['it', 'this', 'that', 'these', 'those', 'them', 'they'],
            'demonstratives': ['the above', 'the previous', 'the mentioned', 'the same'],
            'references': ['earlier', 'before', 'previously', 'last time']
        }
        
        # Intent classification patterns
        self.intent_patterns = {
            QueryIntent.CLARIFICATION: [
                r'\b(what do you mean|clarify|explain|confused|unclear)\b',
                r'\b(can you elaborate|tell me more|go deeper)\b',
                r'\?(.*explain.*|.*mean.*|.*clarify.*)'
            ],
            QueryIntent.ELABORATION: [
                r'\b(more details?|tell me more|elaborate|expand)\b',
                r'\b(go deeper|dig deeper|more information)\b',
                r'\b(what else|anything else|continue)\b'
            ],
            QueryIntent.FOLLOW_UP: [
                r'\b(what about|how about|what happens|and then)\b',
                r'\b(next step|after that|following|subsequent)\b',
                r'\b(also|additionally|furthermore|moreover)\b'
            ]
        }
        
        # Common entity patterns for extraction
        self.entity_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Title case phrases
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w+_\w+\b',  # Snake_case identifiers
            r'\b\w+Process\b|\b\w+Service\b|\b\w+Manager\b',  # Common suffixes
        ]
    
    def rewrite_query(self, 
                     raw_query: str,
                     session: SessionState,
                     history: ConversationHistory,
                     preferences: UserPreferences) -> str:
        """
        Rewrite a raw query to be contextually rich.
        
        Args:
            raw_query: User's original query
            session: Current session state
            history: Conversation history
            preferences: User preferences
            
        Returns:
            Rewritten query with context
        """
        query = raw_query.strip()
        
        try:
            # Step 1: Resolve pronouns and references
            query = self._resolve_pronouns(query, session, history)
            
            # Step 2: Add diagram context if available
            query = self._add_diagram_context(query, session)
            
            # Step 3: Expand with topic context
            query = self._add_topic_context(query, preferences, history)
            
            # Step 4: Adjust for user expertise level
            query = self._adjust_for_expertise(query, preferences)
            
            logger.debug(f"Rewrote query: '{raw_query}' -> '{query}'")
            
            return query
            
        except Exception as e:
            logger.error(f"Error rewriting query '{raw_query}': {e}")
            return raw_query  # Fallback to original
    
    def _resolve_pronouns(self, 
                         query: str,
                         session: SessionState,
                         history: ConversationHistory) -> str:
        """
        Resolve pronouns and references using context.
        
        Args:
            query: Query to process
            session: Session state with active entities
            history: Recent conversation history
            
        Returns:
            Query with resolved references
        """
        query_lower = query.lower()
        
        # Check if query contains context indicators
        has_context_indicators = any(
            any(indicator in query_lower for indicator in indicators)
            for indicators in self.context_indicators.values()
        )
        
        if not has_context_indicators:
            return query
        
        # Get context from session and history
        context_entities = []
        
        # Use active entities from session
        if session.active_entities:
            context_entities.extend(session.active_entities[-3:])  # Most recent 3
        
        # Use entities from recent history
        recent_exchanges = history.get_recent_exchanges(2)
        for exchange in recent_exchanges:
            entities = self.extract_entities(exchange.query_text)
            context_entities.extend(entities[:2])  # Top 2 from each
        
        # Remove duplicates while preserving order
        context_entities = list(dict.fromkeys(context_entities))
        
        if not context_entities:
            return query
        
        # Replace pronouns with context
        replacements = {
            'it': context_entities[0] if context_entities else 'it',
            'this': f"the {context_entities[0]}" if context_entities else 'this',
            'that': f"the {context_entities[0]}" if context_entities else 'that',
            'them': f"the {', '.join(context_entities[:2])}" if len(context_entities) >= 2 else 'them',
            'they': f"the {', '.join(context_entities[:2])}" if len(context_entities) >= 2 else 'they'
        }
        
        # Apply replacements (case-sensitive)
        modified_query = query
        for pronoun, replacement in replacements.items():
            # Replace whole words only
            pattern = r'\b' + re.escape(pronoun) + r'\b'
            modified_query = re.sub(pattern, replacement, modified_query, flags=re.IGNORECASE)
        
        return modified_query
    
    def _add_diagram_context(self, query: str, session: SessionState) -> str:
        """
        Add diagram context to query if available.
        
        Args:
            query: Query to enhance
            session: Session with diagram IDs
            
        Returns:
            Query with diagram context
        """
        if not session.last_retrieved_diagram_ids:
            return query
        
        # Get the most recent diagram ID
        recent_diagram = session.last_retrieved_diagram_ids[-1]
        
        # Check if query already mentions diagrams
        if 'diagram' in query.lower() or recent_diagram in query:
            return query
        
        # Add diagram context
        enhanced_query = f"{query} (in the context of diagram {recent_diagram})"
        
        return enhanced_query
    
    def _add_topic_context(self, 
                          query: str,
                          preferences: UserPreferences,
                          history: ConversationHistory) -> str:
        """
        Add topic context based on user interests and conversation.
        
        Args:
            query: Query to enhance
            preferences: User preferences with topic interests
            history: Conversation history
            
        Returns:
            Query with topic context
        """
        # Get user's topic interests
        user_topics = set(preferences.topic_interest_profile)
        
        # Get recent conversation topics
        recent_topics = set()
        recent_exchanges = history.get_recent_exchanges(3)
        for exchange in recent_exchanges:
            entities = self.extract_entities(exchange.query_text)
            recent_topics.update(entities[:3])
        
        # Find relevant topics for current query
        query_entities = set(self.extract_entities(query))
        
        # Combine all topic sources
        all_topics = user_topics | recent_topics
        
        # Find topics that might be relevant but not mentioned
        relevant_topics = []
        for topic in all_topics:
            if topic.lower() not in query.lower() and len(relevant_topics) < 2:
                # Simple relevance check - can be made more sophisticated
                if any(word in topic.lower() for word in query.lower().split()):
                    relevant_topics.append(topic)
        
        # Add relevant topics if any found
        if relevant_topics:
            topic_context = f" (related to {', '.join(relevant_topics)})"
            return query + topic_context
        
        return query
    
    def _adjust_for_expertise(self, query: str, preferences: UserPreferences) -> str:
        """
        Adjust query based on user's expertise level.
        
        Args:
            query: Query to adjust
            preferences: User preferences
            
        Returns:
            Expertise-adjusted query
        """
        expertise = preferences.expertise_level
        
        # Add expertise context hints
        if 'ExpertiseLevel.BEGINNER' in str(expertise):
            if 'explain' not in query.lower():
                query += " (explain in simple terms)"
        elif 'ExpertiseLevel.EXPERT' in str(expertise):
            if 'technical' not in query.lower() and 'details' not in query.lower():
                query += " (include technical details)"
        
        return query
    
    def classify_intent(self, query: str, session: SessionState) -> QueryIntent:
        """
        Classify the intent of a user query.
        
        Args:
            query: User query to classify
            session: Session state for context
            
        Returns:
            Classified query intent
        """
        query_lower = query.lower()
        
        try:
            # Check each intent pattern
            for intent, patterns in self.intent_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, query_lower):
                        return intent
            
            # Additional context-based classification
            
            # If there are active entities and query is short/vague, likely follow-up
            if session.active_entities and len(query.split()) < 5:
                context_indicators = ['it', 'this', 'that', 'what about']
                if any(indicator in query_lower for indicator in context_indicators):
                    return QueryIntent.FOLLOW_UP
            
            # If asking for "more" or "details", it's elaboration
            if any(word in query_lower for word in ['more', 'details', 'elaborate', 'expand']):
                return QueryIntent.ELABORATION
            
            # Default to new query if no patterns match
            return QueryIntent.NEW_QUERY
            
        except Exception as e:
            logger.error(f"Error classifying intent for query '{query}': {e}")
            return QueryIntent.NEW_QUERY
    
    def analyze_expertise_signals(self, query: str, session: SessionState) -> Dict[str, bool]:
        """
        Analyze query for expertise level signals.
        
        Args:
            query: User query to analyze
            session: Current session state
            
        Returns:
            Dict with boolean signals for expertise learning
        """
        query_lower = query.lower()
        
        return {
            'has_technical_terms': self._has_technical_terms(query),
            'is_clarification_request': self._is_clarification_request(query_lower),
            'is_complex_query': self._is_complex_query(query, query_lower),
            'is_follow_up_question': self._is_follow_up_question(query_lower, session)
        }
    
    def _has_technical_terms(self, query: str) -> bool:
        """Check if query contains technical terminology."""
        # Common technical terms across domains
        technical_terms = {
            'api', 'database', 'sql', 'json', 'xml', 'http', 'https', 'rest', 'graphql',
            'authentication', 'authorization', 'oauth', 'jwt', 'token', 'encryption',
            'algorithm', 'regex', 'cache', 'redis', 'mongodb', 'postgresql', 'mysql',
            'kubernetes', 'docker', 'container', 'microservice', 'pipeline', 'cicd',
            'endpoint', 'middleware', 'framework', 'library', 'dependency', 'npm',
            'repository', 'git', 'branch', 'commit', 'merge', 'deployment', 'staging',
            'production', 'environment', 'configuration', 'logging', 'monitoring',
            'metrics', 'scalability', 'load balancing', 'cdn', 'ssl', 'tls'
        }
        
        query_words = set(query.lower().split())
        return len(query_words.intersection(technical_terms)) > 0
    
    def _is_clarification_request(self, query_lower: str) -> bool:
        """Check if query is asking for clarification or explanation."""
        clarification_patterns = [
            r'\bwhat\s+(does|is|are|do)\b',
            r'\bhow\s+(does|do|can|to)\b', 
            r'\bwhy\s+(does|do|is|are)\b',
            r'\bexplain\b',
            r'\bwhat\s+do\s+you\s+mean\b',
            r'\bcan\s+you\s+(explain|clarify|tell\s+me)\b',
            r'\bi\s+don\'?t\s+understand\b',
            r'\bwhat\'?s\s+the\s+difference\b',
            r'\bwhat\s+exactly\b',
            r'\bin\s+simple\s+terms\b'
        ]
        
        return any(re.search(pattern, query_lower) for pattern in clarification_patterns)
    
    def _is_complex_query(self, query: str, query_lower: str) -> bool:
        """Check if query shows deep understanding or complexity."""
        # Length-based complexity
        if len(query.split()) < 8:
            return False
        
        # Complexity indicators
        complexity_indicators = [
            r'\b(implement|architecture|design|pattern|best\s+practice)\b',
            r'\b(performance|optimization|scalability|security)\b', 
            r'\b(integration|configuration|deployment|monitoring)\b',
            r'\b(compared\s+to|vs|versus|difference\s+between)\b',
            r'\b(pros\s+and\s+cons|advantages|disadvantages)\b',
            r'\b(when\s+should|which\s+approach|alternative)\b'
        ]
        
        # Multi-part questions (and, or, but)
        has_conjunctions = len(re.findall(r'\b(and|or|but|however|also)\b', query_lower)) >= 2
        
        # Has complexity indicators
        has_indicators = any(re.search(pattern, query_lower) for pattern in complexity_indicators)
        
        return has_conjunctions or has_indicators
    
    def _is_follow_up_question(self, query_lower: str, session: SessionState) -> bool:
        """Check if query is a follow-up to previous conversation."""
        if not session.active_entities:
            return False
        
        # Follow-up indicators
        follow_up_patterns = [
            r'\b(what\s+about|how\s+about|and\s+what|also)\b',
            r'\b(furthermore|additionally|moreover|besides)\b',
            r'\b(related\s+to\s+that|regarding\s+that|about\s+that)\b'
        ]
        
        # Pronoun references
        has_pronouns = any(word in query_lower for word in ['it', 'this', 'that', 'them', 'they'])
        
        # Pattern matching
        has_follow_up_patterns = any(re.search(pattern, query_lower) for pattern in follow_up_patterns)
        
        return has_pronouns or has_follow_up_patterns
    
    def extract_entities(self, text: str) -> List[str]:
        """
        Extract key entities from text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of extracted entities
        """
        if not text:
            return []
        
        entities = []
        
        try:
            # Apply entity extraction patterns
            for pattern in self.entity_patterns:
                matches = re.findall(pattern, text)
                entities.extend(matches)
            
            # Clean and filter entities
            entities = [
                entity.strip() 
                for entity in entities 
                if len(entity.strip()) > 2 and not entity.isdigit()
            ]
            
            # Remove duplicates while preserving order
            unique_entities = []
            seen = set()
            for entity in entities:
                entity_lower = entity.lower()
                if entity_lower not in seen:
                    unique_entities.append(entity)
                    seen.add(entity_lower)
            
            # Limit to reasonable number
            return unique_entities[:10]
            
        except Exception as e:
            logger.error(f"Error extracting entities from text '{text[:50]}...': {e}")
            return []
    
    def get_context_keywords(self, 
                           session: SessionState,
                           history: ConversationHistory,
                           max_keywords: int = 15) -> List[str]:
        """
        Get contextual keywords for search enhancement.
        
        Args:
            session: Session state
            history: Conversation history
            max_keywords: Maximum number of keywords
            
        Returns:
            List of context keywords
        """
        keywords = []
        
        try:
            # Add active entities from session
            if session.active_entities:
                keywords.extend(session.active_entities)
            
            # Add keywords from recent conversation
            recent_exchanges = history.get_recent_exchanges(3)
            for exchange in recent_exchanges:
                # Extract entities from both query and response
                query_entities = self.extract_entities(exchange.query_text)
                response_entities = self.extract_entities(exchange.llm_response)
                keywords.extend(query_entities[:3])
                keywords.extend(response_entities[:2])
            
            # Remove duplicates and limit
            unique_keywords = list(dict.fromkeys(keywords))
            return unique_keywords[:max_keywords]
            
        except Exception as e:
            logger.error(f"Error getting context keywords: {e}")
            return []
    
    def should_use_conversational_context(self, 
                                        query: str,
                                        session: SessionState) -> bool:
        """
        Determine if query should use conversational context.
        
        Args:
            query: User query
            session: Session state
            
        Returns:
            True if context should be used
        """
        query_lower = query.lower()
        
        # Check for context indicators
        has_indicators = any(
            any(indicator in query_lower for indicator in indicators)
            for indicators in self.context_indicators.values()
        )
        
        # Check if session has recent activity
        has_recent_activity = (
            session.active_entities 
            or session.last_retrieved_diagram_ids
            or (session.timestamp and 
                (datetime.now() - session.timestamp).total_seconds() < 300)  # 5 minutes
        )
        
        return has_indicators and has_recent_activity