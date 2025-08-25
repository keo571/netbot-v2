"""
Workflow orchestration for hybrid RAG operations.

Implements the multi-phase retrieval workflows defined in the architecture,
coordinating between vector search, graph retrieval, and response synthesis.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import asyncio
from abc import ABC, abstractmethod

from ...shared import get_logger
from ...services.text_rag import TextRAG, SearchQuery as TextSearchQuery
# Import from the correct GraphRAG location
import sys
from pathlib import Path
# Add the graph_rag directory to path
graph_rag_path = Path(__file__).parent.parent.parent.parent / "graph_rag"
sys.path.insert(0, str(graph_rag_path))

from graph_rag.client import GraphRAG
from ...services.context_manager import ContextManager
from .models import RAGQuery, RAGResponse, RAGContext, ProcessingMode, SourceReference, VisualizationData
from .reliability import ReliabilityManager


class BaseWorkflow(ABC):
    """Base class for RAG workflows."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    @abstractmethod
    async def execute(self, query: RAGQuery) -> Tuple[RAGResponse, RAGContext]:
        """Execute the workflow and return response with context."""
        pass
    
    def _create_source_reference(
        self, 
        source_data: Dict[str, Any], 
        source_type: str
    ) -> SourceReference:
        """Create source reference from raw data."""
        return SourceReference(
            source_id=source_data.get('id', 'unknown'),
            source_type=source_type,
            title=source_data.get('title', 'Unknown Source'),
            content_excerpt=source_data.get('content', '')[:500] + '...',
            relevance_score=source_data.get('score', 0.0),
            document_id=source_data.get('document_id'),
            chunk_id=source_data.get('chunk_id'),
            node_id=source_data.get('node_id'),
            diagram_id=source_data.get('diagram_id')
        )


class HybridRAGWorkflow(BaseWorkflow):
    """
    Implements the complete hybrid RAG workflow from the architecture.
    
    Coordinates vector search, graph retrieval, and response synthesis
    with comprehensive reliability assessment.
    """
    
    def __init__(
        self,
        text_rag: Optional[TextRAG] = None,
        graph_rag: Optional[GraphRAG] = None,
        context_manager: Optional[ContextManager] = None,
        reliability_manager: Optional[ReliabilityManager] = None
    ):
        super().__init__()
        
        # Initialize components
        self.text_rag = text_rag or TextRAG()
        self.graph_rag = graph_rag or GraphRAG()
        self.context_manager = context_manager or ContextManager()
        self.reliability_manager = reliability_manager or ReliabilityManager()
        
        self.logger.info("Hybrid RAG workflow initialized")
    
    async def execute(self, query: RAGQuery) -> Tuple[RAGResponse, RAGContext]:
        """
        Execute the complete hybrid RAG workflow.
        
        Implementation of the architecture's multi-phase retrieval:
        1. Vector search for relevant text chunks
        2. Graph retrieval for structural context
        3. Context assembly and validation
        4. Response synthesis with reliability assessment
        """
        start_time = datetime.now()
        context = RAGContext()
        
        try:
            # Phase 1: Vector Search (Text Retrieval)
            text_results = await self._phase_1_vector_search(query, context)
            
            # Phase 2: Graph Retrieval (Structural Context)
            graph_data = await self._phase_2_graph_retrieval(query, text_results, context)
            
            # Phase 3: Context Assembly and Validation
            assembled_context = await self._phase_3_context_assembly(
                query, text_results, graph_data, context
            )
            
            # Phase 4: Response Synthesis
            response = await self._phase_4_response_synthesis(
                query, assembled_context, context
            )
            
            # Phase 5: Reliability Assessment
            final_response = await self._phase_5_reliability_assessment(
                query, response, assembled_context, context
            )
            
            # Record processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            final_response.processing_time_ms = processing_time
            
            self.logger.info(f"Hybrid RAG workflow completed in {processing_time:.2f}ms")
            return final_response, context
            
        except Exception as e:
            self.logger.error(f"Hybrid RAG workflow failed: {e}")
            return await self._create_error_response(query, str(e), context), context
    
    async def _phase_1_vector_search(
        self, 
        query: RAGQuery, 
        context: RAGContext
    ) -> List[Dict[str, Any]]:
        """Phase 1: Vector search for semantically relevant text chunks."""
        phase_start = datetime.now()
        context.processing_steps.append("Phase 1: Vector Search")
        
        try:
            # Build text search query
            text_query = TextSearchQuery(
                query_text=query.query_text,
                top_k=query.top_k,
                similarity_threshold=query.similarity_threshold,
                categories=query.categories
            )
            
            # Perform vector search
            if query.session_id and self.context_manager:
                # Use conversational search if session available
                from ...services.text_rag.integrations import ContextAwareTextRAG
                context_aware_rag = ContextAwareTextRAG(self.text_rag, self.context_manager)
                search_results = await context_aware_rag.conversational_search(
                    query.query_text, query.session_id, query.top_k, query.similarity_threshold
                )
            else:
                # Standard vector search
                search_results = await self.text_rag.search(
                    query.query_text, query.top_k, query.similarity_threshold
                )
            
            # Convert to standard format
            text_results = []
            for result in search_results.results:
                text_results.append({
                    'id': result.chunk_id,
                    'content': result.content,
                    'title': result.document_title,
                    'score': result.fusion_score,
                    'document_id': result.document_id,
                    'chunk_id': result.chunk_id,
                    'metadata': {
                        'chunk_index': result.chunk_index,
                        'categories': result.categories
                    }
                })
            
            context.text_results = text_results
            phase_time = (datetime.now() - phase_start).total_seconds() * 1000
            context.timing_breakdown['vector_search'] = phase_time
            
            self.logger.info(f"Phase 1 completed: {len(text_results)} text results in {phase_time:.2f}ms")
            return text_results
            
        except Exception as e:
            self.logger.error(f"Phase 1 vector search failed: {e}")
            context.timing_breakdown['vector_search'] = (datetime.now() - phase_start).total_seconds() * 1000
            return []
    
    async def _phase_2_graph_retrieval(
        self, 
        query: RAGQuery, 
        text_results: List[Dict[str, Any]], 
        context: RAGContext
    ) -> List[Dict[str, Any]]:
        """Phase 2: Graph retrieval for structural context."""
        phase_start = datetime.now()
        context.processing_steps.append("Phase 2: Graph Retrieval")
        
        try:
            graph_data = []
            
            # Extract diagram IDs from text results or use specified diagram
            diagram_ids = set()
            if query.diagram_id:
                diagram_ids.add(query.diagram_id)
            
            for result in text_results:
                if 'diagram_id' in result.get('metadata', {}):
                    diagram_ids.add(result['metadata']['diagram_id'])
            
            # Retrieve graph data for each diagram
            for diagram_id in diagram_ids:
                try:
                    # Use GraphRAG for structural queries
                    graph_results = await self.graph_rag.search(
                        query=query.query_text,
                        diagram_id=diagram_id,
                        top_k=query.top_k
                    )
                    
                    # Convert graph results to standard format
                    for result in graph_results.get('results', []):
                        graph_data.append({
                            'id': result.get('node_id', 'unknown'),
                            'name': result.get('name', ''),
                            'type': result.get('type', 'node'),
                            'properties': result.get('properties', {}),
                            'diagram_id': diagram_id,
                            'score': result.get('score', 0.0),
                            'relationships': result.get('relationships', [])
                        })
                        
                except Exception as e:
                    self.logger.warning(f"Graph retrieval failed for diagram {diagram_id}: {e}")
                    continue
            
            context.graph_data = graph_data
            context.diagram_metadata = {'diagram_ids': list(diagram_ids)}
            
            phase_time = (datetime.now() - phase_start).total_seconds() * 1000
            context.timing_breakdown['graph_retrieval'] = phase_time
            
            self.logger.info(f"Phase 2 completed: {len(graph_data)} graph nodes in {phase_time:.2f}ms")
            return graph_data
            
        except Exception as e:
            self.logger.error(f"Phase 2 graph retrieval failed: {e}")
            context.timing_breakdown['graph_retrieval'] = (datetime.now() - phase_start).total_seconds() * 1000
            return []
    
    async def _phase_3_context_assembly(
        self,
        query: RAGQuery,
        text_results: List[Dict[str, Any]],
        graph_data: List[Dict[str, Any]],
        context: RAGContext
    ) -> RAGContext:
        """Phase 3: Context assembly and validation."""
        phase_start = datetime.now()
        context.processing_steps.append("Phase 3: Context Assembly")
        
        try:
            # Get conversation context if available
            if query.session_id and self.context_manager:
                try:
                    conversation_history = self.context_manager.get_conversation_history(
                        query.session_id, limit=5
                    )
                    context.conversation_history = [
                        {
                            'role': msg.role,
                            'content': msg.content,
                            'timestamp': msg.timestamp.isoformat() if msg.timestamp else None
                        }
                        for msg in conversation_history
                    ]
                    
                    # Get user preferences
                    if query.user_id:
                        user = self.context_manager.get_user(query.user_id)
                        if user:
                            context.user_preferences = user.preferences
                            
                except Exception as e:
                    self.logger.warning(f"Failed to get conversation context: {e}")
            
            # Calculate context quality metrics
            context.context_quality_score = self._calculate_context_quality(text_results, graph_data)
            context.cross_modal_consistency = self._calculate_cross_modal_consistency(
                text_results, graph_data
            )
            
            phase_time = (datetime.now() - phase_start).total_seconds() * 1000
            context.timing_breakdown['context_assembly'] = phase_time
            
            self.logger.info(f"Phase 3 completed: Context assembled in {phase_time:.2f}ms")
            return context
            
        except Exception as e:
            self.logger.error(f"Phase 3 context assembly failed: {e}")
            context.timing_breakdown['context_assembly'] = (datetime.now() - phase_start).total_seconds() * 1000
            return context
    
    async def _phase_4_response_synthesis(
        self,
        query: RAGQuery,
        context: RAGContext,
        processing_context: RAGContext
    ) -> RAGResponse:
        """Phase 4: Response synthesis using LLM."""
        phase_start = datetime.now()
        processing_context.processing_steps.append("Phase 4: Response Synthesis")
        
        try:
            # Prepare context for LLM
            context_text = self._prepare_context_for_llm(context)
            
            # Generate response (placeholder - would use actual LLM)
            response_text = await self._synthesize_response_with_llm(
                query.query_text, context_text, query.processing_mode
            )
            
            # Create source references
            sources = []
            
            # Add text sources
            for result in context.text_results:
                sources.append(self._create_source_reference(result, 'document_chunk'))
            
            # Add graph sources
            for node in context.graph_data:
                sources.append(self._create_source_reference(node, 'graph_node'))
            
            # Generate visualizations if requested
            visualizations = []
            if query.include_visualizations and context.graph_data:
                viz = await self._generate_visualizations(context.graph_data)
                if viz:
                    visualizations.append(viz)
            
            # Create initial response
            response = RAGResponse(
                query_id=query.query_id,
                response_text=response_text,
                sources=sources,
                visualizations=visualizations,
                confidence_metrics=None,  # Will be set in Phase 5
                context_used=context,
                processing_mode=query.processing_mode,
                processing_time_ms=0.0  # Will be set later
            )
            
            phase_time = (datetime.now() - phase_start).total_seconds() * 1000
            processing_context.timing_breakdown['response_synthesis'] = phase_time
            
            self.logger.info(f"Phase 4 completed: Response synthesized in {phase_time:.2f}ms")
            return response
            
        except Exception as e:
            self.logger.error(f"Phase 4 response synthesis failed: {e}")
            processing_context.timing_breakdown['response_synthesis'] = (datetime.now() - phase_start).total_seconds() * 1000
            raise e
    
    async def _phase_5_reliability_assessment(
        self,
        query: RAGQuery,
        response: RAGResponse,
        context: RAGContext,
        processing_context: RAGContext
    ) -> RAGResponse:
        """Phase 5: Comprehensive reliability assessment."""
        phase_start = datetime.now()
        processing_context.processing_steps.append("Phase 5: Reliability Assessment")
        
        try:
            # Calculate confidence metrics
            confidence_metrics = self.reliability_manager.assess_response_reliability(
                response.response_text,
                context,
                response.sources,
                query.query_text
            )
            
            response.confidence_metrics = confidence_metrics
            
            # Set quality flags based on confidence
            response.requires_verification = confidence_metrics.overall_confidence < 0.6
            response.has_uncertainties = len(confidence_metrics.information_gaps) > 0
            
            # Generate follow-up suggestions
            response.suggested_follow_ups = self._generate_follow_up_suggestions(
                query.query_text, context, confidence_metrics
            )
            
            phase_time = (datetime.now() - phase_start).total_seconds() * 1000
            processing_context.timing_breakdown['reliability_assessment'] = phase_time
            
            self.logger.info(f"Phase 5 completed: Reliability assessed in {phase_time:.2f}ms")
            return response
            
        except Exception as e:
            self.logger.error(f"Phase 5 reliability assessment failed: {e}")
            processing_context.timing_breakdown['reliability_assessment'] = (datetime.now() - phase_start).total_seconds() * 1000
            return response
    
    # Helper methods
    
    def _calculate_context_quality(
        self, 
        text_results: List[Dict[str, Any]], 
        graph_data: List[Dict[str, Any]]
    ) -> float:
        """Calculate context quality score."""
        if not text_results and not graph_data:
            return 0.0
        
        # Simple heuristic based on result counts and scores
        text_quality = 0.0
        if text_results:
            avg_text_score = sum(r.get('score', 0.0) for r in text_results) / len(text_results)
            text_quality = min(avg_text_score + (len(text_results) * 0.1), 1.0)
        
        graph_quality = 0.0
        if graph_data:
            avg_graph_score = sum(n.get('score', 0.0) for n in graph_data) / len(graph_data)
            graph_quality = min(avg_graph_score + (len(graph_data) * 0.05), 1.0)
        
        # Combined quality with higher weight on text
        combined_quality = (text_quality * 0.7) + (graph_quality * 0.3)
        return combined_quality
    
    def _calculate_cross_modal_consistency(
        self, 
        text_results: List[Dict[str, Any]], 
        graph_data: List[Dict[str, Any]]
    ) -> float:
        """Calculate consistency between text and graph data."""
        if not text_results or not graph_data:
            return 1.0  # Single modality is always consistent
        
        # Extract entities from both modalities
        text_entities = set()
        for result in text_results:
            content = result.get('content', '').lower()
            # Simple entity extraction
            words = content.split()
            text_entities.update([w for w in words if len(w) > 3])
        
        graph_entities = set()
        for node in graph_data:
            name = node.get('name', '').lower()
            if name:
                graph_entities.add(name)
        
        # Calculate overlap
        if not text_entities or not graph_entities:
            return 0.5
        
        intersection = len(text_entities & graph_entities)
        union = len(text_entities | graph_entities)
        
        consistency = intersection / union if union > 0 else 0.0
        return consistency
    
    def _prepare_context_for_llm(self, context: RAGContext) -> str:
        """Prepare context for LLM consumption."""
        context_parts = []
        
        # Add text context
        if context.text_results:
            context_parts.append("=== TEXTUAL CONTEXT ===")
            for i, result in enumerate(context.text_results[:5], 1):  # Limit to top 5
                context_parts.append(f"{i}. {result.get('title', 'Unknown')}")
                context_parts.append(f"   {result.get('content', '')[:300]}...")
                context_parts.append("")
        
        # Add graph context
        if context.graph_data:
            context_parts.append("=== STRUCTURAL CONTEXT ===")
            for i, node in enumerate(context.graph_data[:10], 1):  # Limit to top 10
                context_parts.append(f"{i}. {node.get('name', 'Unknown')} ({node.get('type', 'node')})")
                if node.get('properties'):
                    context_parts.append(f"   Properties: {node['properties']}")
                context_parts.append("")
        
        # Add conversation history if available
        if context.conversation_history:
            context_parts.append("=== CONVERSATION HISTORY ===")
            for msg in context.conversation_history[-3:]:  # Last 3 messages
                context_parts.append(f"{msg['role']}: {msg['content']}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    async def _synthesize_response_with_llm(
        self, 
        query: str, 
        context: str, 
        mode: ProcessingMode
    ) -> str:
        """Synthesize response using LLM (placeholder implementation)."""
        # This would integrate with actual LLM service
        # For now, return a structured response based on available context
        
        if not context.strip():
            return f"I don't have enough information to answer '{query}'. Please try a different query or provide more context."
        
        # Simple response generation based on mode
        if mode == ProcessingMode.FAST:
            response = f"Based on the available information, here's a quick answer to '{query}':\n\n{context[:500]}..."
        elif mode == ProcessingMode.COMPREHENSIVE:
            response = f"Here's a comprehensive analysis of '{query}':\n\n{context[:1500]}..."
        else:  # BALANCED or INTERACTIVE
            response = f"Regarding '{query}', here's what I found:\n\n{context[:1000]}..."
        
        return response
    
    async def _generate_visualizations(
        self, 
        graph_data: List[Dict[str, Any]]
    ) -> Optional[VisualizationData]:
        """Generate visualizations from graph data."""
        if not graph_data:
            return None
        
        try:
            # Create nodes and relationships for visualization
            nodes = []
            relationships = []
            
            for node in graph_data:
                nodes.append({
                    'id': node.get('id'),
                    'name': node.get('name'),
                    'type': node.get('type'),
                    'properties': node.get('properties', {})
                })
                
                # Add relationships
                for rel in node.get('relationships', []):
                    relationships.append({
                        'source': node.get('id'),
                        'target': rel.get('target_id'),
                        'type': rel.get('type'),
                        'properties': rel.get('properties', {})
                    })
            
            return VisualizationData(
                visualization_type="network_graph",
                title="Related Network Components",
                nodes=nodes,
                relationships=relationships,
                layout="force_directed",
                style_config={
                    "node_color": "category",
                    "edge_width": "weight",
                    "layout_iterations": 100
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate visualizations: {e}")
            return None
    
    def _generate_follow_up_suggestions(
        self, 
        query: str, 
        context: RAGContext, 
        confidence_metrics
    ) -> List[str]:
        """Generate follow-up suggestions based on query and context."""
        suggestions = []
        
        # Based on confidence level
        if confidence_metrics.overall_confidence < 0.5:
            suggestions.append("Try rephrasing your query with more specific terms")
        
        # Based on available data types
        if context.text_results and not context.graph_data:
            suggestions.append("Would you like to see related network diagrams?")
        elif context.graph_data and not context.text_results:
            suggestions.append("Would you like to see related documentation?")
        
        # Based on query complexity
        if len(query.split()) < 5:
            suggestions.append("Consider asking a more detailed question for better results")
        
        return suggestions[:3]  # Limit to top 3 suggestions
    
    async def _create_error_response(
        self, 
        query: RAGQuery, 
        error_message: str, 
        context: RAGContext
    ) -> RAGResponse:
        """Create error response when workflow fails."""
        from .reliability import ConfidenceCalculator
        
        fallback_metrics = ConfidenceCalculator()._create_fallback_metrics(0)
        
        return RAGResponse(
            query_id=query.query_id,
            response_text=f"I encountered an error processing your query: {error_message}. Please try again or contact support if the issue persists.",
            sources=[],
            visualizations=[],
            confidence_metrics=fallback_metrics,
            context_used=context,
            processing_mode=query.processing_mode,
            processing_time_ms=0.0,
            requires_verification=True,
            has_uncertainties=True
        )


class DocumentWorkflow(BaseWorkflow):
    """Workflow for document processing and ingestion."""
    
    def __init__(self, text_rag: Optional[TextRAG] = None):
        super().__init__()
        self.text_rag = text_rag or TextRAG()
    
    async def execute(self, query: RAGQuery) -> Tuple[RAGResponse, RAGContext]:
        """Execute document processing workflow."""
        # Implementation for document ingestion
        pass


class QueryWorkflow(BaseWorkflow):
    """Workflow for query processing and optimization."""
    
    def __init__(self, context_manager: Optional[ContextManager] = None):
        super().__init__()
        self.context_manager = context_manager or ContextManager()
    
    async def execute(self, query: RAGQuery) -> Tuple[RAGResponse, RAGContext]:
        """Execute query optimization workflow."""
        # Implementation for query enhancement
        pass