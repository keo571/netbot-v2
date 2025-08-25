"""
Reliability and confidence assessment for RAG responses.

Implements comprehensive reliability metrics and hallucination reduction
through evidence validation and consistency checking.
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from datetime import datetime
import re

from ...shared import get_logger
from .models import ConfidenceMetrics, RAGContext, ReliabilityLevel, SourceReference


class ConfidenceCalculator:
    """
    Calculates multi-dimensional confidence scores for RAG responses.
    
    Implements the reliability framework from the architecture document
    with evidence validation and consistency checking.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Confidence calculation weights (simplified)
        self.confidence_weights = {
            'source_coverage': 0.4,
            'response_grounding': 0.4,
            'semantic_consistency': 0.2
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            'high_confidence': 0.8,
            'medium_confidence': 0.5,
            'low_confidence': 0.2
        }
    
    def calculate_response_confidence(
        self, 
        response: str, 
        context: RAGContext,
        sources: List[SourceReference],
        query: str
    ) -> ConfidenceMetrics:
        """
        Calculate comprehensive confidence metrics for a RAG response.
        
        Args:
            response: Generated response text
            context: Assembled context used for generation
            sources: Source references used
            query: Original query
            
        Returns:
            Comprehensive confidence metrics
        """
        try:
            start_time = datetime.now()
            
            # Calculate individual confidence scores (simplified)
            scores = {
                'source_coverage': self._calculate_source_coverage(response, sources, query),
                'response_grounding': self._validate_response_evidence(response, sources),
                'semantic_consistency': self._check_semantic_coherence(response)
            }
            
            # Calculate weighted composite confidence
            overall_confidence = self._calculate_weighted_confidence(scores)
            reliability_level = self._determine_reliability_level(overall_confidence)
            
            # Determine if there are information gaps
            has_gaps = self._has_information_gaps(context, query, scores)
            
            return ConfidenceMetrics(
                overall_confidence=overall_confidence,
                reliability_level=reliability_level,
                source_coverage=scores['source_coverage'],
                response_grounding=scores['response_grounding'],
                has_gaps=has_gaps,
                sources_used=len(sources)
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence metrics: {e}")
            return self._create_fallback_metrics(len(sources))
    
    def _calculate_source_coverage(
        self, 
        response: str, 
        sources: List[SourceReference],
        query: str
    ) -> float:
        """Calculate how well sources cover the query and response."""
        if not sources:
            return 0.0
        
        try:
            # Extract key terms from query and response
            query_terms = self._extract_key_terms(query)
            response_terms = self._extract_key_terms(response)
            
            # Calculate coverage from sources
            total_coverage = 0.0
            for source in sources:
                source_terms = self._extract_key_terms(source.content_excerpt)
                
                # Coverage of query terms
                query_coverage = len(set(query_terms) & set(source_terms)) / max(len(query_terms), 1)
                
                # Coverage of response terms
                response_coverage = len(set(response_terms) & set(source_terms)) / max(len(response_terms), 1)
                
                # Weighted by source relevance score
                source_coverage = (query_coverage + response_coverage) / 2 * source.relevance_score
                total_coverage += source_coverage
            
            # Normalize by number of sources
            return min(total_coverage / len(sources), 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating source coverage: {e}")
            return 0.5
    
    
    def _validate_response_evidence(self, response: str, sources: List[SourceReference]) -> float:
        """Validate how well the response is grounded in provided evidence."""
        if not sources:
            return 0.0
        
        try:
            response_claims = self._extract_claims(response)
            if not response_claims:
                return 0.8  # Simple response, likely well-grounded
            
            grounded_claims = 0
            for claim in response_claims:
                for source in sources:
                    if self._claim_supported_by_source(claim, source.content_excerpt):
                        grounded_claims += 1
                        break
            
            grounding_score = grounded_claims / len(response_claims)
            return grounding_score
            
        except Exception as e:
            self.logger.error(f"Error validating response evidence: {e}")
            return 0.6
    
    def _check_semantic_coherence(self, response: str) -> float:
        """Check internal semantic consistency of the response."""
        try:
            # Simple coherence checks
            sentences = re.split(r'[.!?]+', response.strip())
            if len(sentences) < 2:
                return 0.9  # Single sentence, likely coherent
            
            # Check for contradictory statements (basic heuristics)
            contradiction_indicators = [
                ('not', 'is'), ('never', 'always'), ('no', 'yes'),
                ('cannot', 'can'), ('impossible', 'possible')
            ]
            
            coherence_issues = 0
            text_lower = response.lower()
            
            for neg, pos in contradiction_indicators:
                if neg in text_lower and pos in text_lower:
                    # Simple distance check - if contradictory terms are close, it's likely an issue
                    neg_pos = text_lower.find(neg)
                    pos_pos = text_lower.find(pos)
                    if abs(neg_pos - pos_pos) < 100:  # Within 100 characters
                        coherence_issues += 1
            
            # Penalize based on coherence issues
            coherence_score = max(1.0 - (coherence_issues * 0.2), 0.0)
            return coherence_score
            
        except Exception as e:
            self.logger.error(f"Error checking semantic coherence: {e}")
            return 0.8
    
    def _calculate_weighted_confidence(self, scores: Dict[str, float]) -> float:
        """Calculate weighted composite confidence score."""
        weighted_sum = sum(
            scores[metric] * weight 
            for metric, weight in self.confidence_weights.items()
        )
        return min(max(weighted_sum, 0.0), 1.0)
    
    def _determine_reliability_level(self, confidence: float) -> ReliabilityLevel:
        """Determine reliability level based on confidence score."""
        if confidence >= self.quality_thresholds['high_confidence']:
            return ReliabilityLevel.HIGH
        elif confidence >= self.quality_thresholds['medium_confidence']:
            return ReliabilityLevel.MEDIUM
        elif confidence >= self.quality_thresholds['low_confidence']:
            return ReliabilityLevel.LOW
        else:
            return ReliabilityLevel.UNCERTAIN
    
    def _has_information_gaps(
        self, 
        context: RAGContext, 
        query: str, 
        scores: Dict[str, float]
    ) -> bool:
        """Determine if there are significant information gaps (simplified)."""
        
        # Critical gaps that indicate incomplete information
        if scores['source_coverage'] < 0.3:
            return True
            
        if not context.text_results and not context.graph_data:
            return True  # No sources at all
            
        if scores['response_grounding'] < 0.4:
            return True  # Response not well-supported
            
        return False
    
    
    def _create_fallback_metrics(self, num_sources: int) -> ConfidenceMetrics:
        """Create fallback metrics when calculation fails."""
        base_confidence = min(num_sources * 0.1, 0.5)  # Based on source count
        
        return ConfidenceMetrics(
            overall_confidence=base_confidence,
            reliability_level=ReliabilityLevel.UNCERTAIN,
            source_coverage=base_confidence,
            response_grounding=0.5,
            has_gaps=True,  # Assume gaps when calculation fails
            sources_used=num_sources
        )
    
    # Utility methods for text analysis
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text."""
        # Simple keyword extraction (would use NLP in production)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'she', 'use', 'her', 'how', 'man', 'new', 'now', 'old', 'see', 'get', 'has', 'him', 'his', 'how', 'its'}
        return [word for word in words if word not in stop_words and len(word) > 3]
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        # Simple entity extraction (would use NLP in production)
        # Look for capitalized words and technical terms
        entities = re.findall(r'\b[A-Z][a-zA-Z0-9_-]*\b', text)
        return list(set(entities))
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from response text."""
        # Simple claim extraction - split by sentences
        sentences = re.split(r'[.!?]+', text.strip())
        claims = [s.strip() for s in sentences if len(s.strip()) > 10]
        return claims
    
    def _claim_supported_by_source(self, claim: str, source_content: str) -> bool:
        """Check if a claim is supported by source content."""
        # Simple overlap check (would use semantic similarity in production)
        claim_terms = set(self._extract_key_terms(claim))
        source_terms = set(self._extract_key_terms(source_content))
        
        overlap = len(claim_terms & source_terms)
        threshold = max(len(claim_terms) * 0.3, 1)  # At least 30% overlap
        
        return overlap >= threshold
    
    def _assess_query_complexity(self, query: str) -> float:
        """Assess query complexity (0-1 scale)."""
        # Simple heuristics
        word_count = len(query.split())
        question_words = len(re.findall(r'\b(what|when|where|who|why|how)\b', query.lower()))
        technical_terms = len(re.findall(r'\b(server|network|database|firewall|router|switch)\b', query.lower()))
        
        complexity = min((word_count * 0.05) + (question_words * 0.2) + (technical_terms * 0.15), 1.0)
        return complexity
    
    def _assess_available_information(self, context: RAGContext) -> float:
        """Assess amount of available information (0-1 scale)."""
        info_score = 0.0
        
        # Text results
        if context.text_results:
            text_content = sum(len(result.get('content', '')) for result in context.text_results)
            info_score += min(text_content / 1000, 0.5)  # Up to 0.5 for text
        
        # Graph data
        if context.graph_data:
            graph_nodes = len(context.graph_data)
            info_score += min(graph_nodes / 20, 0.3)  # Up to 0.3 for graph
        
        # Conversation history
        if context.conversation_history:
            info_score += 0.2  # Bonus for context
        
        return min(info_score, 1.0)


class ReliabilityManager:
    """
    Manages overall system reliability and quality assurance.
    
    Orchestrates confidence calculation, quality monitoring,
    and reliability reporting across the RAG system.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.confidence_calculator = ConfidenceCalculator()
        
        # System reliability tracking
        self.reliability_stats = {
            'total_queries': 0,
            'high_confidence_responses': 0,
            'medium_confidence_responses': 0,
            'low_confidence_responses': 0,
            'uncertain_responses': 0,
            'avg_confidence': 0.0
        }
    
    def assess_response_reliability(
        self,
        response: str,
        context: RAGContext,
        sources: List[SourceReference],
        query: str
    ) -> ConfidenceMetrics:
        """
        Assess response reliability and update system statistics.
        
        Args:
            response: Generated response text
            context: Context used for generation
            sources: Sources used
            query: Original query
            
        Returns:
            Comprehensive reliability assessment
        """
        try:
            # Calculate confidence metrics
            metrics = self.confidence_calculator.calculate_response_confidence(
                response, context, sources, query
            )
            
            # Update system statistics
            self._update_reliability_stats(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error assessing response reliability: {e}")
            return self.confidence_calculator._create_fallback_metrics(len(sources))
    
    def _update_reliability_stats(self, metrics: ConfidenceMetrics) -> None:
        """Update system-wide reliability statistics."""
        self.reliability_stats['total_queries'] += 1
        
        # Update confidence distribution
        if metrics.reliability_level == ReliabilityLevel.HIGH:
            self.reliability_stats['high_confidence_responses'] += 1
        elif metrics.reliability_level == ReliabilityLevel.MEDIUM:
            self.reliability_stats['medium_confidence_responses'] += 1
        elif metrics.reliability_level == ReliabilityLevel.LOW:
            self.reliability_stats['low_confidence_responses'] += 1
        else:
            self.reliability_stats['uncertain_responses'] += 1
        
        # Update average confidence (running average)
        total = self.reliability_stats['total_queries']
        current_avg = self.reliability_stats['avg_confidence']
        self.reliability_stats['avg_confidence'] = (
            (current_avg * (total - 1) + metrics.overall_confidence) / total
        )
    
    def get_system_reliability_report(self) -> Dict[str, Any]:
        """Get comprehensive system reliability report."""
        total = self.reliability_stats['total_queries']
        
        if total == 0:
            return {
                'status': 'No queries processed yet',
                'reliability_distribution': {},
                'average_confidence': 0.0,
                'system_health': 'Unknown'
            }
        
        distribution = {
            'high_confidence': self.reliability_stats['high_confidence_responses'] / total,
            'medium_confidence': self.reliability_stats['medium_confidence_responses'] / total,
            'low_confidence': self.reliability_stats['low_confidence_responses'] / total,
            'uncertain': self.reliability_stats['uncertain_responses'] / total
        }
        
        # Determine system health
        high_confidence_ratio = distribution['high_confidence']
        avg_confidence = self.reliability_stats['avg_confidence']
        
        if high_confidence_ratio >= 0.7 and avg_confidence >= 0.8:
            system_health = 'Excellent'
        elif high_confidence_ratio >= 0.5 and avg_confidence >= 0.6:
            system_health = 'Good'
        elif high_confidence_ratio >= 0.3 and avg_confidence >= 0.4:
            system_health = 'Fair'
        else:
            system_health = 'Needs Attention'
        
        return {
            'total_queries_processed': total,
            'reliability_distribution': distribution,
            'average_confidence': avg_confidence,
            'system_health': system_health,
            'recommendations': self._generate_system_recommendations(distribution, avg_confidence)
        }
    
    def _generate_system_recommendations(
        self, 
        distribution: Dict[str, float], 
        avg_confidence: float
    ) -> List[str]:
        """Generate system improvement recommendations."""
        recommendations = []
        
        if distribution['uncertain'] > 0.2:
            recommendations.append("High uncertain response rate - review source quality")
        
        if avg_confidence < 0.6:
            recommendations.append("Low average confidence - consider expanding knowledge base")
        
        if distribution['high_confidence'] < 0.4:
            recommendations.append("Low high-confidence rate - improve retrieval algorithms")
        
        return recommendations