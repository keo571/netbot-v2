"""
Query enhancement and rewriting using conversation context.

Enhances user queries by resolving pronouns, adding context,
and maintaining topic continuity across conversations.
"""

import re
from typing import List, Dict, Any, Optional, Set, Tuple

from ....shared import get_logger, get_model_client
from ..models import Session, Message, User, ContextQuery


class QueryRewriter:
    """
    Enhances user queries using conversation context.
    
    Capabilities:
    - Resolves pronouns and vague references
    - Adds diagram context to queries
    - Maintains topic continuity
    - Handles follow-up questions
    """
    
    def __init__(self):
        """Initialize the query rewriter."""
        self.logger = get_logger(__name__)
        self.model_client = get_model_client()
        
        # Patterns for pronoun and reference resolution
        self.pronoun_patterns = [
            r'\\b(it|this|that|these|those|they|them)\\b',
            r'\\b(the same|similar|like that|such)\\b',
            r'\\b(again|also|too|as well)\\b'
        ]
        
        # Vague query patterns that need context
        self.vague_patterns = [
            r'^(show|find|get|what|where|how)\\s+(me\\s+)?(that|this|it|them|those)',
            r'^(explain|describe|tell me about)\\s+(that|this|it)',
            r'^(what about|how about|and)\\s+',
            r'^(more|details|info)\\s+(on|about)?',
            r'\\?\\s*$'  # Questions ending with just a question mark
        ]
    
    def rewrite_with_context(self,
                           raw_query: str,
                           session: Session,
                           user: User,
                           conversation_history: List[Message] = None) -> ContextQuery:
        """
        Enhance a query using conversation context.
        
        Args:
            raw_query: Original user query
            session: Current session context
            user: User preferences and profile
            conversation_history: Recent conversation messages
            
        Returns:
            Enhanced query with context and metadata
        """
        try:
            enhanced_query = raw_query.strip()
            enhancements_applied = []
            entities_resolved = {}
            context_added = []
            
            # Check if query needs enhancement
            needs_enhancement = self._needs_enhancement(raw_query)
            
            if needs_enhancement:
                # Step 1: Resolve pronouns and references
                enhanced_query, pronoun_resolutions = self._resolve_pronouns(
                    enhanced_query, session, conversation_history or []
                )
                if pronoun_resolutions:
                    enhancements_applied.append("pronoun_resolution")
                    entities_resolved.update(pronoun_resolutions)
                
                # Step 2: Add diagram context if relevant
                if session.diagram_id and self._should_add_diagram_context(enhanced_query):
                    enhanced_query, diagram_context = self._add_diagram_context(
                        enhanced_query, session
                    )
                    if diagram_context:
                        enhancements_applied.append("diagram_context")
                        context_added.append(f"diagram: {session.diagram_id}")
                
                # Step 3: Add entity context from active discussion
                if session.active_entities:
                    enhanced_query, entity_context = self._add_entity_context(
                        enhanced_query, session
                    )
                    if entity_context:
                        enhancements_applied.append("entity_context")
                        context_added.extend(entity_context)
                
                # Step 4: Handle follow-up patterns
                if self._is_follow_up_query(raw_query) and conversation_history:
                    enhanced_query, follow_up_context = self._handle_follow_up(
                        enhanced_query, conversation_history
                    )
                    if follow_up_context:
                        enhancements_applied.append("follow_up_context")
                        context_added.extend(follow_up_context)
                
                # Step 5: Apply user preference context
                enhanced_query = self._apply_user_context(enhanced_query, user)
            
            # Calculate enhancement confidence
            enhancement_confidence = self._calculate_enhancement_confidence(
                raw_query, enhanced_query, enhancements_applied
            )
            
            # Calculate context relevance
            context_relevance = self._calculate_context_relevance(
                enhanced_query, session, conversation_history or []
            )
            
            result = ContextQuery(
                original_query=raw_query,
                enhanced_query=enhanced_query,
                session_id=session.session_id,
                enhancements_applied=enhancements_applied,
                entities_resolved=entities_resolved,
                context_added=context_added,
                enhancement_confidence=enhancement_confidence,
                context_relevance=context_relevance
            )
            
            if result.was_enhanced:
                self.logger.info(f"Enhanced query for session {session.session_id}: {raw_query} -> {enhanced_query}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Query rewriting failed: {e}")
            # Return original query with error context
            return ContextQuery(
                original_query=raw_query,
                enhanced_query=raw_query,
                session_id=session.session_id,
                enhancement_confidence=0.0,
                context_relevance=0.0
            )
    
    def _needs_enhancement(self, query: str) -> bool:
        """Determine if a query needs contextual enhancement."""
        query_lower = query.lower().strip()
        
        # Check for vague patterns
        for pattern in self.vague_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return True
        
        # Check for pronoun usage
        for pattern in self.pronoun_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return True
        
        # Check if query is very short and possibly incomplete
        if len(query.split()) <= 2 and not query.endswith('?'):
            return True
        
        return False
    
    def _resolve_pronouns(self,
                         query: str,
                         session: Session,
                         conversation_history: List[Message]) -> Tuple[str, Dict[str, str]]:
        """Resolve pronouns and vague references in the query."""
        enhanced_query = query
        resolutions = {}
        
        try:
            # Get recent entities from conversation
            recent_entities = self._extract_recent_entities(conversation_history, limit=5)
            
            # Combine with active session entities
            all_entities = list(session.active_entities) + recent_entities
            
            if not all_entities:
                return enhanced_query, resolutions
            
            # Simple pronoun resolution patterns
            pronoun_mappings = {
                r'\\bit\\b': all_entities[0] if all_entities else None,
                r'\\bthis\\b': all_entities[0] if all_entities else None,
                r'\\bthat\\b': all_entities[1] if len(all_entities) > 1 else all_entities[0] if all_entities else None,
                r'\\bthey\\b': ', '.join(all_entities[:2]) if len(all_entities) >= 2 else None,
                r'\\bthem\\b': ', '.join(all_entities[:3]) if len(all_entities) >= 2 else None,
            }
            
            # Apply resolutions
            for pronoun_pattern, replacement in pronoun_mappings.items():
                if replacement and re.search(pronoun_pattern, enhanced_query, re.IGNORECASE):
                    # Only replace if it makes sense contextually
                    if self._is_reasonable_replacement(enhanced_query, pronoun_pattern, replacement):
                        old_query = enhanced_query
                        enhanced_query = re.sub(
                            pronoun_pattern, 
                            replacement, 
                            enhanced_query, 
                            count=1, 
                            flags=re.IGNORECASE
                        )
                        
                        if old_query != enhanced_query:
                            pronoun = re.search(pronoun_pattern, old_query, re.IGNORECASE)
                            if pronoun:
                                resolutions[pronoun.group()] = replacement
            
            return enhanced_query, resolutions
            
        except Exception as e:
            self.logger.error(f"Pronoun resolution failed: {e}")
            return query, {}
    
    def _add_diagram_context(self, query: str, session: Session) -> Tuple[str, bool]:
        """Add diagram context to the query if relevant."""
        if not session.diagram_id:
            return query, False
        
        # Check if diagram is already mentioned
        if session.diagram_id.lower() in query.lower():
            return query, False
        
        # Add diagram context for certain query types
        context_keywords = ['find', 'show', 'locate', 'search', 'get', 'what', 'where', 'how']
        
        query_lower = query.lower()
        for keyword in context_keywords:
            if query_lower.startswith(keyword):
                enhanced_query = f"{query} in diagram {session.diagram_id}"
                return enhanced_query, True
        
        return query, False
    
    def _add_entity_context(self, query: str, session: Session) -> Tuple[str, List[str]]:
        """Add context from currently active entities."""
        if not session.active_entities:
            return query, []
        
        context_added = []
        enhanced_query = query
        
        # Check if we should add entity context
        context_indicators = ['related', 'connected', 'similar', 'other', 'more', 'also']
        
        query_lower = query.lower()
        should_add_context = any(indicator in query_lower for indicator in context_indicators)
        
        if should_add_context:
            # Add context about currently discussed entities
            entities_list = ', '.join(list(session.active_entities)[:3])
            enhanced_query = f"{query} (related to: {entities_list})"
            context_added = [f"active_entity: {entity}" for entity in list(session.active_entities)[:3]]
        
        return enhanced_query, context_added
    
    def _is_follow_up_query(self, query: str) -> bool:
        """Check if query is a follow-up to previous conversation."""
        follow_up_patterns = [
            r'^(and|also|what about|how about|plus)',
            r'^(more|additional|extra|further)',
            r'^(details|info|information)\\s+(on|about)?',
            r'^(yes|no|ok|sure|right),?\\s+',
            r'\\?\\s*$'
        ]
        
        query_lower = query.lower().strip()
        return any(re.search(pattern, query_lower) for pattern in follow_up_patterns)
    
    def _handle_follow_up(self, query: str, conversation_history: List[Message]) -> Tuple[str, List[str]]:
        """Handle follow-up queries by adding context from previous messages."""
        if not conversation_history:
            return query, []
        
        context_added = []
        enhanced_query = query
        
        # Get the last user query for context
        last_user_query = None
        for msg in reversed(conversation_history):
            if msg.message_type.value == "user_query":
                last_user_query = msg
                break
        
        if last_user_query:
            # Extract key terms from the last query
            last_query_terms = self._extract_key_terms(last_user_query.content)
            
            if last_query_terms:
                # Add context from previous query
                terms_text = ', '.join(last_query_terms[:3])
                enhanced_query = f"{query} (continuing from: {terms_text})"
                context_added = [f"previous_query: {term}" for term in last_query_terms[:3]]
        
        return enhanced_query, context_added
    
    def _apply_user_context(self, query: str, user: User) -> str:
        """Apply user preferences and expertise level to query."""
        # This is a simple implementation - could be enhanced with ML
        
        # Add expertise context for very technical queries
        if user.expertise_level.value == "beginner" and self._is_technical_query(query):
            return f"{query} (explain for beginners)"
        elif user.expertise_level.value == "expert" and not self._is_technical_query(query):
            return f"{query} (technical details)"
        
        return query
    
    def _should_add_diagram_context(self, query: str) -> bool:
        """Determine if diagram context should be added to the query."""
        # Don't add if diagram is already mentioned
        diagram_keywords = ['diagram', 'chart', 'image', 'figure']
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in diagram_keywords):
            return False
        
        # Add for search/find type queries
        search_keywords = ['find', 'show', 'locate', 'search', 'get', 'what', 'where']
        return any(query_lower.startswith(keyword) for keyword in search_keywords)
    
    def _is_reasonable_replacement(self, query: str, pronoun_pattern: str, replacement: str) -> bool:
        """Check if a pronoun replacement makes contextual sense."""
        # Simple heuristics to avoid bad replacements
        
        # Don't replace if replacement is too long
        if len(replacement) > 50:
            return False
        
        # Don't replace pronouns in certain contexts
        bad_contexts = [
            r'\\bif\\s+' + pronoun_pattern,  # "if it"
            r'\\bwhen\\s+' + pronoun_pattern,  # "when it"
            r'\\bwhere\\s+' + pronoun_pattern,  # "where it"
        ]
        
        for context in bad_contexts:
            if re.search(context, query, re.IGNORECASE):
                return False
        
        return True
    
    def _extract_recent_entities(self, conversation_history: List[Message], limit: int = 5) -> List[str]:
        """Extract entities mentioned in recent conversation."""
        entities = []
        
        for msg in reversed(conversation_history[-10:]):  # Look at last 10 messages
            if msg.entities_mentioned:
                entities.extend(msg.entities_mentioned)
            
            if len(entities) >= limit:
                break
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)
        
        return unique_entities[:limit]
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text."""
        # Simple keyword extraction
        import re
        
        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        
        # Split into words and filter
        words = re.findall(r'\\b\\w+\\b', text.lower())
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        return key_terms[:5]  # Return top 5 key terms
    
    def _is_technical_query(self, query: str) -> bool:
        """Determine if a query is technical in nature."""
        technical_terms = [
            'protocol', 'network', 'server', 'database', 'configuration',
            'architecture', 'topology', 'firewall', 'router', 'switch',
            'api', 'interface', 'connection', 'port', 'security'
        ]
        
        query_lower = query.lower()
        return any(term in query_lower for term in technical_terms)
    
    def _calculate_enhancement_confidence(self,
                                        original_query: str,
                                        enhanced_query: str,
                                        enhancements: List[str]) -> float:
        """Calculate confidence score for the enhancement."""
        if original_query == enhanced_query:
            return 1.0  # No enhancement needed
        
        # Base confidence on types of enhancements applied
        confidence_weights = {
            'pronoun_resolution': 0.8,
            'diagram_context': 0.9,
            'entity_context': 0.7,
            'follow_up_context': 0.6,
        }
        
        if not enhancements:
            return 0.5  # Default confidence
        
        # Calculate weighted average
        total_weight = sum(confidence_weights.get(enhancement, 0.5) for enhancement in enhancements)
        return min(total_weight / len(enhancements), 1.0)
    
    def _calculate_context_relevance(self,
                                   enhanced_query: str,
                                   session: Session,
                                   conversation_history: List[Message]) -> float:
        """Calculate how relevant the context is to the query."""
        relevance_score = 0.0
        
        # Check diagram relevance
        if session.diagram_id and session.diagram_id.lower() in enhanced_query.lower():
            relevance_score += 0.3
        
        # Check entity relevance
        if session.active_entities:
            query_lower = enhanced_query.lower()
            relevant_entities = sum(1 for entity in session.active_entities if entity.lower() in query_lower)
            relevance_score += min(relevant_entities * 0.2, 0.4)
        
        # Check conversation continuity
        if conversation_history:
            last_msg = conversation_history[-1]
            if last_msg.intent and last_msg.intent in enhanced_query.lower():
                relevance_score += 0.3
        
        return min(relevance_score, 1.0)