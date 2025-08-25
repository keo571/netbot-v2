"""
Result reranking for TextRAG search.

Provides advanced reranking algorithms to improve search result
relevance and quality through multiple scoring strategies.
"""

from typing import List, Optional, Dict, Any
import numpy as np

from ....shared import get_logger
from ..models import SearchResult


class ResultReranker:
    """
    Advanced result reranking system.
    
    Applies multiple reranking strategies to improve search result
    relevance, diversity, and overall quality.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.logger.info("Result Reranker initialized")
    
    async def rerank_results(self, 
                           results: List[SearchResult], 
                           query_text: str,
                           session_id: Optional[str] = None) -> List[SearchResult]:
        """
        Rerank search results using multiple strategies.
        
        Args:
            results: Initial search results
            query_text: Original query text
            session_id: Session ID for context-aware reranking
            
        Returns:
            Reranked results
        """
        if len(results) <= 1:
            return results
        
        try:
            # Apply multiple reranking strategies
            reranked = results.copy()
            
            # 1. Query-document relevance boosting
            reranked = self._apply_relevance_boosting(reranked, query_text)
            
            # 2. Diversity-based reranking
            reranked = self._apply_diversity_reranking(reranked)
            
            # 3. Quality-based scoring
            reranked = self._apply_quality_scoring(reranked)
            
            # 4. Context-aware boosting (if session provided)
            if session_id:
                reranked = await self._apply_context_reranking(reranked, session_id)
            
            # 5. Final score computation and sorting
            reranked = self._compute_final_scores(reranked)
            
            # Update final ranks
            for i, result in enumerate(reranked):
                result.final_rank = i + 1
            
            self.logger.debug(f"Reranked {len(results)} results")
            return reranked
            
        except Exception as e:
            self.logger.error(f"Reranking failed: {e}")
            return results
    
    def _apply_relevance_boosting(self, results: List[SearchResult], query_text: str) -> List[SearchResult]:
        """Apply query-document relevance boosting."""
        query_terms = set(query_text.lower().split())
        
        for result in results:
            content_words = set(result.content.lower().split())
            
            # Calculate term overlap
            overlap = len(query_terms & content_words)
            total_query_terms = len(query_terms)
            
            if total_query_terms > 0:
                term_coverage = overlap / total_query_terms
                
                # Boost fusion score based on term coverage
                relevance_boost = term_coverage * 0.2  # Up to 20% boost
                result.fusion_score = min(1.0, result.fusion_score + relevance_boost)
            
            # Boost for exact phrase matches
            if query_text.lower() in result.content.lower():
                result.fusion_score = min(1.0, result.fusion_score + 0.15)
            
            # Boost for keyword matches
            keyword_matches = sum(1 for kw in result.keywords 
                                 if any(term in kw.lower() for term in query_terms))
            
            if result.keywords and keyword_matches > 0:
                keyword_boost = (keyword_matches / len(result.keywords)) * 0.1
                result.fusion_score = min(1.0, result.fusion_score + keyword_boost)
        
        return results
    
    def _apply_diversity_reranking(self, results: List[SearchResult]) -> List[SearchResult]:
        """Apply diversity-based reranking to reduce redundancy."""
        if len(results) <= 2:
            return results
        
        # Group results by document
        doc_groups = {}
        for result in results:
            if result.document_id not in doc_groups:
                doc_groups[result.document_id] = []
            doc_groups[result.document_id].append(result)
        
        # If all results are from different documents, no diversity penalty needed
        if len(doc_groups) == len(results):
            return results
        
        # Apply diversity penalty for multiple results from same document
        for doc_id, doc_results in doc_groups.items():
            if len(doc_results) > 1:
                # Sort by original fusion score
                doc_results.sort(key=lambda r: r.fusion_score, reverse=True)
                
                # Apply increasing penalty to lower-ranked results from same document
                for i, result in enumerate(doc_results):
                    if i > 0:  # Don't penalize the top result from each document
                        diversity_penalty = 0.1 * i  # Increasing penalty
                        result.fusion_score = max(0.0, result.fusion_score - diversity_penalty)
        
        return results
    
    def _apply_quality_scoring(self, results: List[SearchResult]) -> List[SearchResult]:
        """Apply content quality-based scoring."""
        for result in results:
            quality_score = 0.0
            
            # Length-based quality (optimal length around 500-1000 chars)
            content_length = len(result.content)
            if content_length < 50:
                length_quality = content_length / 50.0
            elif content_length <= 1000:
                length_quality = 1.0
            else:
                length_quality = max(0.3, 1000.0 / content_length)
            
            quality_score += length_quality * 0.3
            
            # Structure-based quality
            sentences = result.content.split('.')
            if sentences:
                avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
                # Optimal sentence length around 15-20 words
                if 10 <= avg_sentence_length <= 25:
                    structure_quality = 1.0
                else:
                    structure_quality = max(0.5, 1.0 - abs(avg_sentence_length - 17.5) / 17.5)
                
                quality_score += structure_quality * 0.2
            
            # Keyword richness
            if result.keywords:
                keyword_quality = min(1.0, len(result.keywords) / 10.0)  # Normalize to max 10 keywords
                quality_score += keyword_quality * 0.2
            
            # Entity richness
            if result.named_entities:
                entity_quality = min(1.0, len(result.named_entities) / 5.0)  # Normalize to max 5 entities
                quality_score += entity_quality * 0.15
            
            # Topic coherence
            if result.topics:
                topic_quality = min(1.0, len(result.topics) / 3.0)  # Normalize to max 3 topics
                quality_score += topic_quality * 0.15
            
            # Apply quality boost (up to 10% boost)
            quality_boost = (quality_score / 5.0) * 0.1  # Normalize and scale
            result.fusion_score = min(1.0, result.fusion_score + quality_boost)
        
        return results
    
    async def _apply_context_reranking(self, results: List[SearchResult], session_id: str) -> List[SearchResult]:
        """Apply context-aware reranking based on conversation history."""
        try:
            # This would integrate with Context Manager to get conversation history
            # For now, apply simplified context boosting
            
            # Placeholder: boost results that contain recently mentioned entities
            # In real implementation, this would:
            # 1. Get conversation history from Context Manager
            # 2. Extract entities and topics from recent messages
            # 3. Boost results that align with conversation context
            
            return results
            
        except Exception as e:
            self.logger.error(f"Context reranking failed: {e}")
            return results
    
    def _compute_final_scores(self, results: List[SearchResult]) -> List[SearchResult]:
        """Compute final scores and sort results."""
        # Normalize fusion scores to 0-1 range
        if results:
            max_score = max(r.fusion_score for r in results)
            min_score = min(r.fusion_score for r in results)
            
            if max_score > min_score:
                for result in results:
                    normalized_score = (result.fusion_score - min_score) / (max_score - min_score)
                    result.fusion_score = normalized_score
        
        # Sort by fusion score
        results.sort(key=lambda r: r.fusion_score, reverse=True)
        
        return results
    
    def get_reranking_explanation(self, result: SearchResult, query_text: str) -> Dict[str, Any]:
        """Get explanation of reranking factors for a result."""
        explanation = {
            'original_similarity': result.similarity_score,
            'lexical_score': result.lexical_score,
            'final_fusion_score': result.fusion_score,
            'boosting_factors': []
        }
        
        # Analyze boosting factors
        query_terms = set(query_text.lower().split())
        content_words = set(result.content.lower().split())
        overlap = len(query_terms & content_words)
        
        if overlap > 0:
            explanation['boosting_factors'].append({
                'factor': 'term_overlap',
                'value': overlap / len(query_terms),
                'description': f'Query terms found in content: {overlap}/{len(query_terms)}'
            })
        
        if query_text.lower() in result.content.lower():
            explanation['boosting_factors'].append({
                'factor': 'exact_phrase_match',
                'value': 0.15,
                'description': 'Exact query phrase found in content'
            })
        
        if result.keywords:
            keyword_matches = sum(1 for kw in result.keywords 
                                 if any(term in kw.lower() for term in query_terms))
            if keyword_matches > 0:
                explanation['boosting_factors'].append({
                    'factor': 'keyword_match',
                    'value': keyword_matches / len(result.keywords),
                    'description': f'Keywords matching query: {keyword_matches}/{len(result.keywords)}'
                })
        
        return explanation