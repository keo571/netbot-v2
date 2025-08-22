"""
Retrieval filtering and re-ranking for context-aware search results.

Filters and ranks search results based on user preferences, session context,
and conversation history to provide the most relevant results for the user's
current needs and expertise level.
"""

import logging
import math
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime

try:
    from ..models import SessionState, UserPreferences, ExpertiseLevel, ResponseStyle
except ImportError:
    from models import SessionState, UserPreferences, ExpertiseLevel, ResponseStyle


logger = logging.getLogger(__name__)


class RetrievalFilter:
    """
    Handles filtering and re-ranking of retrieval results based on context.
    """
    
    def __init__(self):
        """Initialize the retrieval filter."""
        
        # Expertise level indicators
        self.expertise_indicators = {
            ExpertiseLevel.BEGINNER: {
                'preferred_terms': ['overview', 'introduction', 'basics', 'simple', 'guide'],
                'avoid_terms': ['advanced', 'complex', 'implementation', 'technical details'],
                'content_types': ['tutorial', 'guide', 'overview']
            },
            ExpertiseLevel.INTERMEDIATE: {
                'preferred_terms': ['how to', 'steps', 'process', 'workflow', 'example'],
                'avoid_terms': ['basic', 'introduction', 'extremely technical'],
                'content_types': ['howto', 'process', 'example']
            },
            ExpertiseLevel.EXPERT: {
                'preferred_terms': ['implementation', 'technical', 'advanced', 'details', 'specification'],
                'avoid_terms': ['basic', 'introduction', 'simple'],
                'content_types': ['reference', 'specification', 'implementation']
            }
        }
        
        # Response style preferences
        self.style_preferences = {
            ResponseStyle.CONCISE: {
                'preferred_length': 'short',
                'avoid_terms': ['detailed', 'comprehensive', 'extensive'],
                'boost_terms': ['summary', 'brief', 'overview']
            },
            ResponseStyle.DETAILED: {
                'preferred_length': 'long', 
                'avoid_terms': ['summary', 'brief'],
                'boost_terms': ['detailed', 'comprehensive', 'extensive', 'thorough']
            },
            ResponseStyle.BALANCED: {
                'preferred_length': 'medium',
                'boost_terms': ['example', 'explanation', 'overview']
            }
        }
    
    def filter_by_preferences(self,
                            results: List[Dict[str, Any]],
                            preferences: UserPreferences) -> List[Dict[str, Any]]:
        """
        Filter results based on user preferences.
        
        Args:
            results: Raw retrieval results
            preferences: User preferences
            
        Returns:
            Filtered results
        """
        if not results:
            return results
        
        try:
            filtered_results = []
            
            expertise_config = self.expertise_indicators.get(preferences.expertise_level, {})
            style_config = self.style_preferences.get(preferences.response_style, {})
            
            for result in results:
                # Calculate preference score
                score = self._calculate_preference_score(result, expertise_config, style_config)
                
                # Add preference score to result
                result = result.copy()
                result['preference_score'] = score
                result['original_score'] = result.get('score', 0.0)
                
                # Filter out results that don't meet minimum threshold
                if score > 0.3:  # Minimum threshold
                    filtered_results.append(result)
            
            logger.debug(f"Filtered {len(results)} results to {len(filtered_results)} based on preferences")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error filtering results by preferences: {e}")
            return results  # Return original on error
    
    def rank_by_session_context(self,
                               results: List[Dict[str, Any]],
                               session: SessionState) -> List[Dict[str, Any]]:
        """
        Re-rank results based on session context.
        
        Args:
            results: Results to rank
            session: Session state with context
            
        Returns:
            Re-ranked results
        """
        if not results:
            return results
        
        try:
            # Calculate context scores
            for result in results:
                context_score = self._calculate_context_score(result, session)
                
                # Combine with existing scores
                original_score = result.get('original_score', result.get('score', 0.0))
                preference_score = result.get('preference_score', 0.5)
                
                # Weighted combination
                final_score = (
                    original_score * 0.4 +
                    preference_score * 0.3 +
                    context_score * 0.3
                )
                
                result['context_score'] = context_score
                result['final_score'] = final_score
            
            # Sort by final score
            ranked_results = sorted(results, key=lambda x: x.get('final_score', 0), reverse=True)
            
            logger.debug(f"Re-ranked {len(results)} results using session context")
            
            return ranked_results
            
        except Exception as e:
            logger.error(f"Error ranking results by session context: {e}")
            return results
    
    def _calculate_preference_score(self,
                                  result: Dict[str, Any],
                                  expertise_config: Dict[str, Any],
                                  style_config: Dict[str, Any]) -> float:
        """
        Calculate how well a result matches user preferences.
        
        Args:
            result: Search result to score
            expertise_config: Expertise level configuration
            style_config: Response style configuration
            
        Returns:
            Preference score (0.0 to 1.0)
        """
        score = 0.5  # Base score
        
        # Get text content from result
        content = self._extract_content(result)
        content_lower = content.lower()
        
        # Expertise level scoring
        if expertise_config:
            preferred_terms = expertise_config.get('preferred_terms', [])
            avoid_terms = expertise_config.get('avoid_terms', [])
            
            # Boost for preferred terms
            for term in preferred_terms:
                if term in content_lower:
                    score += 0.1
            
            # Penalty for terms to avoid
            for term in avoid_terms:
                if term in content_lower:
                    score -= 0.15
        
        # Response style scoring  
        if style_config:
            boost_terms = style_config.get('boost_terms', [])
            avoid_terms = style_config.get('avoid_terms', [])
            
            # Boost for style-preferred terms
            for term in boost_terms:
                if term in content_lower:
                    score += 0.05
            
            # Penalty for style-inappropriate terms
            for term in avoid_terms:
                if term in content_lower:
                    score -= 0.1
        
        # Content length preference
        if style_config:
            preferred_length = style_config.get('preferred_length')
            content_length = len(content.split())
            
            if preferred_length == 'short' and content_length < 100:
                score += 0.1
            elif preferred_length == 'long' and content_length > 200:
                score += 0.1
            elif preferred_length == 'medium' and 50 < content_length < 200:
                score += 0.1
        
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    
    def _calculate_context_score(self,
                               result: Dict[str, Any],
                               session: SessionState) -> float:
        """
        Calculate how well a result matches session context.
        
        Args:
            result: Search result to score
            session: Session state
            
        Returns:
            Context score (0.0 to 1.0)
        """
        score = 0.3  # Base score
        
        content = self._extract_content(result)
        content_lower = content.lower()
        
        # Active entities scoring
        if session.active_entities:
            entity_matches = 0
            for entity in session.active_entities:
                if entity.lower() in content_lower:
                    entity_matches += 1
            
            # Boost based on entity match ratio
            entity_ratio = entity_matches / len(session.active_entities)
            score += entity_ratio * 0.3
        
        # Diagram context scoring
        if session.last_retrieved_diagram_ids:
            diagram_id = session.last_retrieved_diagram_ids[-1]  # Most recent
            
            # Check if result mentions the diagram
            if 'diagram_id' in result:
                if result['diagram_id'] == diagram_id:
                    score += 0.2
                elif result['diagram_id'] in session.last_retrieved_diagram_ids:
                    score += 0.1
            
            # Check content for diagram references
            if diagram_id in content:
                score += 0.15
        
        # Recency scoring - prefer content related to recent activity
        if session.timestamp:
            time_since_activity = (datetime.now() - session.timestamp).total_seconds()
            
            # Higher score for results if session is very recent
            if time_since_activity < 300:  # 5 minutes
                score += 0.1
            elif time_since_activity < 1800:  # 30 minutes
                score += 0.05
        
        return max(0.0, min(1.0, score))
    
    def _extract_content(self, result: Dict[str, Any]) -> str:
        """
        Extract text content from a search result.
        
        Args:
            result: Search result dictionary
            
        Returns:
            Combined text content
        """
        content_parts = []
        
        # Common content fields
        content_fields = ['content', 'text', 'body', 'description', 'title', 'summary']
        
        for field in content_fields:
            if field in result and result[field]:
                content_parts.append(str(result[field]))
        
        return ' '.join(content_parts)
    
    def filter_by_relevance_threshold(self,
                                    results: List[Dict[str, Any]],
                                    min_score: float = 0.4) -> List[Dict[str, Any]]:
        """
        Filter results by minimum relevance score.
        
        Args:
            results: Results to filter
            min_score: Minimum score threshold
            
        Returns:
            Filtered results
        """
        return [
            result for result in results 
            if result.get('final_score', result.get('score', 0)) >= min_score
        ]
    
    def limit_results(self,
                     results: List[Dict[str, Any]],
                     max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Limit the number of results returned.
        
        Args:
            results: Results to limit
            max_results: Maximum number of results
            
        Returns:
            Limited results
        """
        return results[:max_results]
    
    def diversify_results(self,
                         results: List[Dict[str, Any]],
                         diversity_field: str = 'source',
                         max_per_source: int = 3) -> List[Dict[str, Any]]:
        """
        Diversify results to avoid too many from the same source.
        
        Args:
            results: Results to diversify
            diversity_field: Field to use for diversification
            max_per_source: Maximum results per source
            
        Returns:
            Diversified results
        """
        if not results:
            return results
        
        try:
            diversified = []
            source_counts = {}
            
            for result in results:
                source = result.get(diversity_field, 'unknown')
                current_count = source_counts.get(source, 0)
                
                if current_count < max_per_source:
                    diversified.append(result)
                    source_counts[source] = current_count + 1
            
            logger.debug(f"Diversified {len(results)} results to {len(diversified)}")
            
            return diversified
            
        except Exception as e:
            logger.error(f"Error diversifying results: {e}")
            return results
    
    def boost_recent_content(self,
                           results: List[Dict[str, Any]],
                           boost_factor: float = 0.1) -> List[Dict[str, Any]]:
        """
        Boost scores for more recent content.
        
        Args:
            results: Results to boost
            boost_factor: Amount to boost recent content
            
        Returns:
            Results with boosted scores
        """
        if not results:
            return results
        
        try:
            now = datetime.now()
            
            for result in results:
                # Check for timestamp fields
                timestamp_fields = ['timestamp', 'created_at', 'updated_at', 'date']
                
                for field in timestamp_fields:
                    if field in result and result[field]:
                        try:
                            # Parse timestamp (assuming ISO format)
                            if isinstance(result[field], str):
                                timestamp = datetime.fromisoformat(result[field])
                            else:
                                timestamp = result[field]
                            
                            # Calculate age in days
                            age_days = (now - timestamp).days
                            
                            # Apply decay boost (more recent = higher boost)
                            if age_days < 7:  # Within a week
                                boost = boost_factor * (1 - age_days / 7)
                                current_score = result.get('final_score', result.get('score', 0))
                                result['final_score'] = current_score + boost
                            
                            break  # Only use first valid timestamp
                            
                        except (ValueError, TypeError):
                            continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error boosting recent content: {e}")
            return results
    
    def apply_all_filters(self,
                         results: List[Dict[str, Any]],
                         preferences: UserPreferences,
                         session: SessionState,
                         max_results: int = 10,
                         min_score: float = 0.4) -> List[Dict[str, Any]]:
        """
        Apply complete filtering and ranking pipeline.
        
        Args:
            results: Raw search results
            preferences: User preferences
            session: Session state
            max_results: Maximum results to return
            min_score: Minimum relevance score
            
        Returns:
            Filtered, ranked, and limited results
        """
        if not results:
            return results
        
        try:
            # Step 1: Filter by preferences
            filtered = self.filter_by_preferences(results, preferences)
            
            # Step 2: Re-rank by session context
            ranked = self.rank_by_session_context(filtered, session)
            
            # Step 3: Boost recent content
            boosted = self.boost_recent_content(ranked)
            
            # Step 4: Filter by minimum score
            min_filtered = self.filter_by_relevance_threshold(boosted, min_score)
            
            # Step 5: Diversify results
            diversified = self.diversify_results(min_filtered)
            
            # Step 6: Limit results
            limited = self.limit_results(diversified, max_results)
            
            logger.info(f"Applied complete filter pipeline: {len(results)} -> {len(limited)} results")
            
            return limited
            
        except Exception as e:
            logger.error(f"Error in complete filtering pipeline: {e}")
            return results[:max_results]  # Fallback