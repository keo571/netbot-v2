"""
Context-based filtering and re-ranking of search results.

Filters and re-ranks search results based on user preferences,
conversation context, and relevance to the current discussion.
"""

from typing import List, Dict, Any, Optional, Tuple
import math

from ....shared import get_logger
from ..models import Session, User, ExpertiseLevel, ResponseStyle


class RetrievalFilter:
    """
    Filters and re-ranks search results based on context.
    
    Capabilities:
    - Filters results by user expertise level
    - Re-ranks based on conversation context
    - Applies user topic preferences
    - Removes duplicate or irrelevant results
    """
    
    def __init__(self):
        """Initialize the retrieval filter."""
        self.logger = get_logger(__name__)
        
        # Expertise level mappings for complexity filtering
        self.complexity_thresholds = {
            ExpertiseLevel.BEGINNER: 0.3,
            ExpertiseLevel.INTERMEDIATE: 0.6,
            ExpertiseLevel.ADVANCED: 0.8,
            ExpertiseLevel.EXPERT: 1.0
        }
    
    def filter_by_context(self,
                         search_results: List[Dict[str, Any]],
                         user: User,
                         session: Session,
                         relevance_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        Filter and re-rank search results based on context.
        
        Args:
            search_results: Raw search results to filter
            user: User preferences and profile
            session: Current session context
            relevance_threshold: Minimum relevance score to keep
            
        Returns:
            Filtered and re-ranked results
        """
        try:
            if not search_results:
                return []
            
            # Step 1: Apply basic quality filters
            filtered_results = self._apply_quality_filters(search_results)
            
            # Step 2: Filter by user expertise level
            filtered_results = self._filter_by_expertise(filtered_results, user)
            
            # Step 3: Apply user preference filters
            filtered_results = self._apply_user_preferences(filtered_results, user)
            
            # Step 4: Apply session context filtering
            filtered_results = self._apply_session_context(filtered_results, session)
            
            # Step 5: Calculate contextual relevance scores
            scored_results = self._calculate_contextual_scores(
                filtered_results, user, session
            )
            
            # Step 6: Filter by relevance threshold
            relevant_results = [
                result for result in scored_results 
                if result.get('contextual_relevance', 0) >= relevance_threshold
            ]
            
            # Step 7: Sort by contextual relevance
            final_results = sorted(
                relevant_results,
                key=lambda x: x.get('contextual_relevance', 0),
                reverse=True
            )
            
            # Step 8: Remove duplicates
            final_results = self._remove_duplicates(final_results)
            
            self.logger.debug(
                f"Filtered {len(search_results)} results to {len(final_results)} "
                f"for user {user.user_id}"
            )
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Result filtering failed: {e}")
            # Return original results if filtering fails
            return search_results
    
    def re_rank_by_preferences(self,
                             results: List[Dict[str, Any]],
                             user: User,
                             boost_factor: float = 0.3) -> List[Dict[str, Any]]:
        """
        Re-rank results based on user preferences and interests.
        
        Args:
            results: Results to re-rank
            user: User profile
            boost_factor: How much to boost preferred content
            
        Returns:
            Re-ranked results
        """
        try:
            if not results:
                return results
            
            # Calculate preference scores for each result
            for result in results:
                preference_score = self._calculate_preference_score(result, user)
                
                # Boost the original relevance score
                original_score = result.get('relevance_score', 0.5)
                boosted_score = original_score + (preference_score * boost_factor)
                
                result['preference_boosted_score'] = min(boosted_score, 1.0)
                result['preference_score'] = preference_score
            
            # Sort by boosted scores
            re_ranked = sorted(
                results,
                key=lambda x: x.get('preference_boosted_score', 0),
                reverse=True
            )
            
            return re_ranked
            
        except Exception as e:
            self.logger.error(f"Re-ranking failed: {e}")
            return results
    
    def _apply_quality_filters(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply basic quality filters to results."""
        filtered = []
        
        for result in results:
            # Filter out results with very low confidence
            confidence = result.get('confidence_score', 0.5)
            if confidence < 0.1:
                continue
            
            # Filter out empty or very short content
            content_fields = ['label', 'content', 'description', 'name']
            has_content = any(
                result.get(field) and len(str(result.get(field, '')).strip()) > 2
                for field in content_fields
            )
            
            if not has_content:
                continue
            
            filtered.append(result)
        
        return filtered
    
    def _filter_by_expertise(self, 
                           results: List[Dict[str, Any]], 
                           user: User) -> List[Dict[str, Any]]:
        """Filter results based on user expertise level."""
        if user.expertise_level == ExpertiseLevel.EXPERT:
            # Experts see everything
            return results
        
        complexity_threshold = self.complexity_thresholds.get(
            user.expertise_level, 0.6
        )
        
        filtered = []
        for result in results:
            # Calculate complexity score
            complexity = self._estimate_content_complexity(result)
            
            # Keep results that match user's expertise level
            if complexity <= complexity_threshold:
                filtered.append(result)
            # Also keep highly relevant results even if complex
            elif result.get('relevance_score', 0) > 0.8:
                filtered.append(result)
        
        return filtered
    
    def _apply_user_preferences(self, 
                              results: List[Dict[str, Any]], 
                              user: User) -> List[Dict[str, Any]]:
        """Apply user topic preferences and learned preferences."""
        if not user.topic_interests and not user.learned_preferences:
            return results
        
        filtered = []
        for result in results:
            # Check topic interests
            if user.topic_interests:
                result_text = self._extract_text_content(result)
                topic_match = any(
                    topic.lower() in result_text.lower()
                    for topic in user.topic_interests
                )
                
                # Keep if matches interests or is highly relevant
                if topic_match or result.get('relevance_score', 0) > 0.7:
                    filtered.append(result)
            else:
                filtered.append(result)
        
        return filtered
    
    def _apply_session_context(self, 
                             results: List[Dict[str, Any]], 
                             session: Session) -> List[Dict[str, Any]]:
        """Filter results based on current session context."""
        if not session.active_entities and not session.diagram_id:
            return results
        
        filtered = []
        for result in results:
            context_relevance = 0.0
            
            # Check diagram context
            if session.diagram_id:
                diagram_match = (
                    result.get('diagram_id') == session.diagram_id or
                    session.diagram_id.lower() in self._extract_text_content(result).lower()
                )
                if diagram_match:
                    context_relevance += 0.5
            
            # Check active entities
            if session.active_entities:
                result_text = self._extract_text_content(result)
                entity_matches = sum(
                    1 for entity in session.active_entities
                    if entity.lower() in result_text.lower()
                )
                context_relevance += min(entity_matches * 0.2, 0.4)
            
            # Keep results with some context relevance or high general relevance
            if context_relevance > 0.1 or result.get('relevance_score', 0) > 0.6:
                result['session_context_score'] = context_relevance
                filtered.append(result)
        
        return filtered
    
    def _calculate_contextual_scores(self,
                                   results: List[Dict[str, Any]],
                                   user: User,
                                   session: Session) -> List[Dict[str, Any]]:
        """Calculate comprehensive contextual relevance scores."""
        for result in results:
            # Base score from original search
            base_score = result.get('relevance_score', 0.5)
            
            # User preference alignment
            preference_score = self._calculate_preference_score(result, user)
            
            # Session context alignment
            session_score = result.get('session_context_score', 0)
            
            # Recency boost if applicable
            recency_score = self._calculate_recency_score(result)
            
            # Combine scores with weights
            contextual_relevance = (
                base_score * 0.4 +           # Base search relevance
                preference_score * 0.3 +     # User preferences
                session_score * 0.2 +        # Session context
                recency_score * 0.1          # Recency
            )
            
            result['contextual_relevance'] = min(contextual_relevance, 1.0)
            result['preference_alignment'] = preference_score
            result['session_alignment'] = session_score
            result['recency_score'] = recency_score
        
        return results
    
    def _calculate_preference_score(self, result: Dict[str, Any], user: User) -> float:
        """Calculate how well a result aligns with user preferences."""
        score = 0.0
        result_text = self._extract_text_content(result)
        
        # Topic interests
        if user.topic_interests:
            topic_matches = sum(
                1 for topic in user.topic_interests
                if topic.lower() in result_text.lower()
            )
            score += min(topic_matches * 0.2, 0.4)
        
        # Frequent entities
        if user.frequent_entities:
            entity_matches = sum(
                1 for entity in user.frequent_entities.keys()
                if entity.lower() in result_text.lower()
            )
            score += min(entity_matches * 0.1, 0.3)
        
        # Response style alignment
        if user.response_style == ResponseStyle.TECHNICAL:
            # Prefer technical content
            technical_terms = ['protocol', 'configuration', 'architecture', 'specification']
            technical_score = sum(
                1 for term in technical_terms
                if term in result_text.lower()
            )
            score += min(technical_score * 0.1, 0.3)
        
        return min(score, 1.0)
    
    def _calculate_recency_score(self, result: Dict[str, Any]) -> float:
        """Calculate recency score for the result."""
        # This is a placeholder - in practice, you'd check timestamps
        # For now, just return a base score
        return 0.1
    
    def _estimate_content_complexity(self, result: Dict[str, Any]) -> float:
        """Estimate the complexity of content for expertise filtering."""
        content = self._extract_text_content(result)
        
        # Simple heuristics for complexity
        complexity_indicators = [
            'protocol', 'configuration', 'architecture', 'specification',
            'implementation', 'algorithm', 'optimization', 'integration',
            'advanced', 'complex', 'sophisticated', 'enterprise'
        ]
        
        # Technical terms density
        word_count = len(content.split())
        if word_count == 0:
            return 0.0
        
        complex_terms = sum(
            1 for indicator in complexity_indicators
            if indicator in content.lower()
        )
        
        complexity = min(complex_terms / word_count * 10, 1.0)
        
        # Adjust based on other factors
        if result.get('type', '').lower() in ['specification', 'protocol', 'standard']:
            complexity += 0.2
        
        return min(complexity, 1.0)
    
    def _extract_text_content(self, result: Dict[str, Any]) -> str:
        """Extract all text content from a result for analysis."""
        content_fields = ['label', 'content', 'description', 'name', 'type', 'properties']
        
        text_parts = []
        for field in content_fields:
            value = result.get(field, '')
            if value:
                if isinstance(value, dict):
                    # Handle nested properties
                    text_parts.extend(str(v) for v in value.values() if v)
                else:
                    text_parts.append(str(value))
        
        return ' '.join(text_parts)
    
    def _remove_duplicates(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on content similarity."""
        if len(results) <= 1:
            return results
        
        unique_results = []
        seen_contents = set()
        
        for result in results:
            # Create a content signature
            content_key = self._create_content_signature(result)
            
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                unique_results.append(result)
        
        return unique_results
    
    def _create_content_signature(self, result: Dict[str, Any]) -> str:
        """Create a signature for duplicate detection."""
        # Use key identifying fields
        signature_fields = ['id', 'label', 'name']
        
        signature_parts = []
        for field in signature_fields:
            if field in result:
                signature_parts.append(str(result[field]).lower().strip())
        
        if not signature_parts:
            # Fallback to content hash
            content = self._extract_text_content(result)
            return str(hash(content[:100]))  # Use first 100 chars
        
        return '|'.join(signature_parts)
    
    def get_filtering_stats(self, 
                          original_results: List[Dict[str, Any]],
                          filtered_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the filtering process."""
        return {
            'original_count': len(original_results),
            'filtered_count': len(filtered_results),
            'filtering_ratio': len(filtered_results) / len(original_results) if original_results else 0,
            'avg_contextual_relevance': sum(
                result.get('contextual_relevance', 0) for result in filtered_results
            ) / len(filtered_results) if filtered_results else 0,
            'avg_preference_alignment': sum(
                result.get('preference_alignment', 0) for result in filtered_results
            ) / len(filtered_results) if filtered_results else 0,
            'avg_session_alignment': sum(
                result.get('session_alignment', 0) for result in filtered_results
            ) / len(filtered_results) if filtered_results else 0
        }