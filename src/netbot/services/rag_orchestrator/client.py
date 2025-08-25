"""
Simple client interface for RAG Orchestrator.

Provides a clean, easy-to-use API for interacting with the hybrid RAG system.
"""

from typing import Dict, List, Any, Optional, Union
import asyncio

from ...shared import get_logger
from .orchestrator import RAGOrchestrator
from .models import (
    RAGQuery, RAGResponse, BatchRAGRequest, BatchRAGResponse,
    ProcessingStatus, QueryType, ProcessingMode
)


class RAGClient:
    """
    Simple client interface for the RAG Orchestrator.
    
    Provides convenient methods for common RAG operations with
    sensible defaults and automatic resource management.
    """
    
    def __init__(self, orchestrator: Optional[RAGOrchestrator] = None):
        self.logger = get_logger(__name__)
        self.orchestrator = orchestrator or RAGOrchestrator()
        self.logger.info("RAG Client initialized")
    
    async def ask(
        self,
        question: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        diagram_id: Optional[str] = None,
        mode: str = "balanced",
        top_k: int = 5,
        include_visualizations: bool = True
    ) -> Dict[str, Any]:
        """
        Ask a question and get a comprehensive answer.
        
        Args:
            question: Natural language question
            session_id: Session ID for conversational context
            user_id: User ID for personalization
            diagram_id: Specific diagram to search
            mode: Processing mode (fast, balanced, comprehensive)
            top_k: Number of results to retrieve
            include_visualizations: Whether to include visualizations
            
        Returns:
            Dictionary with answer and supporting information
        """
        try:
            # Create query
            query = RAGQuery(
                query_text=question,
                query_type=QueryType.HYBRID_FUSION,
                processing_mode=ProcessingMode(mode),
                session_id=session_id,
                user_id=user_id,
                diagram_id=diagram_id,
                top_k=top_k,
                include_visualizations=include_visualizations
            )
            
            # Execute query
            response = await self.orchestrator.query(query)
            
            # Format response for client consumption
            return {
                'answer': response.response_text,
                'confidence': response.confidence_metrics.overall_confidence,
                'reliability': response.confidence_metrics.reliability_level.value,
                'sources': [
                    {
                        'title': source.title,
                        'excerpt': source.content_excerpt,
                        'relevance': source.relevance_score,
                        'type': source.source_type
                    }
                    for source in response.sources
                ],
                'visualizations': [
                    {
                        'type': viz.visualization_type,
                        'title': viz.title,
                        'data': {
                            'nodes': viz.nodes,
                            'relationships': viz.relationships
                        }
                    }
                    for viz in response.visualizations
                ],
                'suggestions': response.suggested_follow_ups,
                'processing_time_ms': response.processing_time_ms,
                'requires_verification': response.requires_verification
            }
            
        except Exception as e:
            self.logger.error(f"Question failed: {e}")
            return {
                'answer': f"I'm sorry, I encountered an error: {str(e)}",
                'confidence': 0.0,
                'reliability': 'uncertain',
                'sources': [],
                'visualizations': [],
                'suggestions': ['Try rephrasing your question', 'Contact support if the issue persists'],
                'processing_time_ms': 0.0,
                'requires_verification': True
            }
    
    async def search(
        self,
        query: str,
        search_type: str = "hybrid",
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Search for relevant information.
        
        Args:
            query: Search query
            search_type: Type of search (semantic, graph, hybrid)
            top_k: Number of results
            similarity_threshold: Minimum similarity score
            categories: Filter by content categories
            
        Returns:
            Search results with metadata
        """
        try:
            # Map search type to query type
            query_type_map = {
                'semantic': QueryType.SEMANTIC_SEARCH,
                'graph': QueryType.GRAPH_TRAVERSAL,
                'hybrid': QueryType.HYBRID_FUSION
            }
            
            query_obj = RAGQuery(
                query_text=query,
                query_type=query_type_map.get(search_type, QueryType.HYBRID_FUSION),
                processing_mode=ProcessingMode.FAST,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                categories=categories,
                include_visualizations=False  # Focus on search results
            )
            
            response = await self.orchestrator.query(query_obj)
            
            return {
                'results': [
                    {
                        'id': source.source_id,
                        'title': source.title,
                        'content': source.content_excerpt,
                        'score': source.relevance_score,
                        'type': source.source_type
                    }
                    for source in response.sources
                ],
                'total_found': len(response.sources),
                'processing_time_ms': response.processing_time_ms,
                'search_quality': response.confidence_metrics.overall_confidence
            }
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return {
                'results': [],
                'total_found': 0,
                'processing_time_ms': 0.0,
                'search_quality': 0.0,
                'error': str(e)
            }
    
    async def add_document(
        self,
        content: str,
        title: str,
        document_type: str = "text",
        categories: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a document to the knowledge base.
        
        Args:
            content: Document content
            title: Document title
            document_type: Type of document
            categories: Document categories
            metadata: Additional metadata
            
        Returns:
            Document addition result
        """
        try:
            doc_metadata = metadata or {}
            if categories:
                doc_metadata['categories'] = categories
            
            result = await self.orchestrator.add_document(
                content=content,
                title=title,
                document_type=document_type,
                metadata=doc_metadata
            )
            
            return {
                'success': True,
                'document_id': result['document_id'],
                'title': result['title'],
                'chunks_created': result.get('chunks_created', 0),
                'status': result.get('status', 'processed')
            }
            
        except Exception as e:
            self.logger.error(f"Document addition failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def process_diagram(
        self,
        image_path: str,
        diagram_id: Optional[str] = None,
        title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a diagram and add it to the knowledge base.
        
        Args:
            image_path: Path to diagram image
            diagram_id: Optional diagram ID (auto-generated if not provided)
            title: Optional diagram title
            
        Returns:
            Diagram processing result
        """
        try:
            if not diagram_id:
                import uuid
                diagram_id = f"diagram_{uuid.uuid4().hex[:12]}"
            
            result = await self.orchestrator.process_diagram(
                image_path=image_path,
                diagram_id=diagram_id,
                title=title
            )
            
            return {
                'success': True,
                'diagram_id': result['diagram_id'],
                'nodes_created': result.get('nodes_created', 0),
                'relationships_created': result.get('relationships_created', 0),
                'processing_time_ms': result.get('processing_time_ms', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Diagram processing failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def start_conversation(
        self,
        user_id: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new conversation session.
        
        Args:
            user_id: Optional user ID
            preferences: User preferences
            
        Returns:
            Session ID for the conversation
        """
        try:
            session_id = self.orchestrator.context_manager.create_session(
                user_id=user_id,
                preferences=preferences or {}
            )
            
            self.logger.info(f"Started conversation session: {session_id}")
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to start conversation: {e}")
            raise e
    
    async def continue_conversation(
        self,
        session_id: str,
        message: str,
        mode: str = "interactive"
    ) -> Dict[str, Any]:
        """
        Continue an existing conversation.
        
        Args:
            session_id: Conversation session ID
            message: User message
            mode: Processing mode
            
        Returns:
            Conversational response
        """
        try:
            # Record user message
            self.orchestrator.context_manager.update_session(
                session_id=session_id,
                query=message,
                response="",  # Will be updated after processing
                retrieved_context=[]
            )
            
            # Process conversational query
            result = await self.ask(
                question=message,
                session_id=session_id,
                mode=mode
            )
            
            # Update session with response
            self.orchestrator.context_manager.update_session(
                session_id=session_id,
                query=message,
                response=result['answer'],
                retrieved_context=[
                    {'source': source['title'], 'relevance': source['relevance']}
                    for source in result['sources']
                ]
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Conversation continuation failed: {e}")
            return {
                'answer': f"I encountered an error: {str(e)}",
                'confidence': 0.0,
                'reliability': 'uncertain',
                'sources': [],
                'visualizations': [],
                'suggestions': ['Try rephrasing your message'],
                'processing_time_ms': 0.0,
                'requires_verification': True
            }
    
    async def batch_ask(
        self,
        questions: List[str],
        session_id: Optional[str] = None,
        parallel: bool = True,
        max_concurrent: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Ask multiple questions in batch.
        
        Args:
            questions: List of questions
            session_id: Optional session ID
            parallel: Whether to process in parallel
            max_concurrent: Maximum concurrent processing
            
        Returns:
            List of answers
        """
        try:
            # Create batch request
            queries = [
                RAGQuery(
                    query_text=question,
                    query_type=QueryType.HYBRID_FUSION,
                    processing_mode=ProcessingMode.BALANCED,
                    session_id=session_id
                )
                for question in questions
            ]
            
            batch_request = BatchRAGRequest(
                queries=queries,
                parallel_processing=parallel,
                max_concurrent=max_concurrent
            )
            
            # Process batch
            batch_response = await self.orchestrator.batch_query(batch_request)
            
            # Format results
            results = []
            for response in batch_response.successful_responses:
                results.append({
                    'question': next(
                        q.query_text for q in queries 
                        if q.query_id == response.query_id
                    ),
                    'answer': response.response_text,
                    'confidence': response.confidence_metrics.overall_confidence,
                    'sources': [
                        {
                            'title': source.title,
                            'relevance': source.relevance_score
                        }
                        for source in response.sources[:3]  # Top 3 sources
                    ]
                })
            
            # Add failed queries
            for failed in batch_response.failed_queries:
                results.append({
                    'question': next(
                        q.query_text for q in queries 
                        if q.query_id == failed['query_id']
                    ),
                    'answer': f"Error: {failed['error']}",
                    'confidence': 0.0,
                    'sources': []
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch ask failed: {e}")
            return [
                {
                    'question': question,
                    'answer': f"Error: {str(e)}",
                    'confidence': 0.0,
                    'sources': []
                }
                for question in questions
            ]
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health and status information."""
        try:
            return self.orchestrator.get_system_status()
        except Exception as e:
            self.logger.error(f"Failed to get system health: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'services': {},
                'performance_stats': {}
            }
    
    async def close(self) -> None:
        """Clean up resources."""
        try:
            self.orchestrator.close()
            self.logger.info("RAG Client closed")
        except Exception as e:
            self.logger.error(f"Error closing RAG Client: {e}")
    
    # Synchronous wrapper methods for convenience
    
    def ask_sync(self, question: str, **kwargs) -> Dict[str, Any]:
        """Synchronous version of ask method."""
        return asyncio.run(self.ask(question, **kwargs))
    
    def search_sync(self, query: str, **kwargs) -> Dict[str, Any]:
        """Synchronous version of search method."""
        return asyncio.run(self.search(query, **kwargs))
    
    def add_document_sync(self, content: str, title: str, **kwargs) -> Dict[str, Any]:
        """Synchronous version of add_document method."""
        return asyncio.run(self.add_document(content, title, **kwargs))