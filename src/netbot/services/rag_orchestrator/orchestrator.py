"""
Main RAG Orchestrator service.

Provides centralized coordination of the hybrid RAG system, implementing
the architecture's multi-service integration patterns with comprehensive
error handling and performance optimization.
"""

from typing import Dict, List, Any, Optional, Union
import asyncio
from datetime import datetime
import uuid

from ...shared import get_logger, get_database
from ...shared.exceptions import ValidationError, ServiceError
from ...services.text_rag import TextRAG
# Import from the correct GraphRAG location
import sys
from pathlib import Path
# Add the graph_rag directory to path
graph_rag_path = Path(__file__).parent.parent.parent.parent / "graph_rag"
sys.path.insert(0, str(graph_rag_path))

from graph_rag.client import GraphRAG
from ...services.context_manager import ContextManager
from .models import (
    RAGQuery, RAGResponse, RAGContext, BatchRAGRequest, BatchRAGResponse,
    ProcessingStatus, QueryType, ProcessingMode
)
from .workflows import HybridRAGWorkflow, DocumentWorkflow, QueryWorkflow
from .reliability import ReliabilityManager


class RAGOrchestrator:
    """
    Main RAG Orchestrator service.
    
    Coordinates all hybrid RAG operations, implementing the architecture's
    multi-phase retrieval workflow with comprehensive reliability assessment.
    """
    
    def __init__(
        self,
        text_rag: Optional[TextRAG] = None,
        graph_rag: Optional[GraphRAG] = None,
        context_manager: Optional[ContextManager] = None,
        reliability_manager: Optional[ReliabilityManager] = None,
        database_client=None
    ):
        self.logger = get_logger(__name__)
        
        # Initialize core components
        self.text_rag = text_rag or TextRAG()
        self.graph_rag = graph_rag or GraphRAG()
        self.context_manager = context_manager or ContextManager()
        self.reliability_manager = reliability_manager or ReliabilityManager()
        
        # Database connection
        self.db_client = database_client or get_database()
        
        # Initialize workflows
        self.hybrid_workflow = HybridRAGWorkflow(
            self.text_rag, self.graph_rag, self.context_manager, self.reliability_manager
        )
        self.document_workflow = DocumentWorkflow(self.text_rag)
        self.query_workflow = QueryWorkflow(self.context_manager)
        
        # Processing status tracking
        self.active_operations: Dict[str, ProcessingStatus] = {}
        
        # Performance metrics
        self.performance_stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'avg_processing_time_ms': 0.0,
            'total_processing_time_ms': 0.0
        }
        
        self.logger.info("RAG Orchestrator initialized successfully")
    
    async def query(
        self, 
        query: Union[RAGQuery, str, Dict[str, Any]], 
        **kwargs
    ) -> RAGResponse:
        """
        Execute a RAG query with full orchestration.
        
        Args:
            query: RAG query (as object, string, or dict)
            **kwargs: Additional query parameters
            
        Returns:
            Complete RAG response with reliability metrics
        """
        start_time = datetime.now()
        
        try:
            # Normalize query input
            rag_query = self._normalize_query_input(query, **kwargs)
            
            # Validate query
            self._validate_query(rag_query)
            
            # Select appropriate workflow
            workflow = self._select_workflow(rag_query)
            
            # Execute workflow
            response, context = await workflow.execute(rag_query)
            
            # Update performance metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_performance_stats(processing_time, True)
            
            # Log successful query
            self.logger.info(
                f"Query {rag_query.query_id} completed successfully "
                f"({processing_time:.2f}ms, confidence: {response.confidence_metrics.overall_confidence:.2f})"
            )
            
            return response
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_performance_stats(processing_time, False)
            
            error_msg = f"Query processing failed: {str(e)}"
            self.logger.error(error_msg)
            
            # Return error response
            if isinstance(query, RAGQuery):
                query_id = query.query_id
            else:
                query_id = f"error_{uuid.uuid4().hex[:8]}"
            
            return await self._create_error_response(query_id, error_msg)
    
    async def query_async(self, query: Union[RAGQuery, str, Dict[str, Any]], **kwargs) -> str:
        """
        Start asynchronous query processing.
        
        Args:
            query: RAG query
            **kwargs: Additional parameters
            
        Returns:
            Operation ID for status tracking
        """
        try:
            # Normalize query
            rag_query = self._normalize_query_input(query, **kwargs)
            operation_id = f"async_{rag_query.query_id}"
            
            # Create processing status
            status = ProcessingStatus(
                operation_id=operation_id,
                status="started",
                current_stage="initialization",
                estimated_completion=None
            )
            self.active_operations[operation_id] = status
            
            # Start background processing
            asyncio.create_task(self._process_async_query(operation_id, rag_query))
            
            self.logger.info(f"Async query {operation_id} started")
            return operation_id
            
        except Exception as e:
            self.logger.error(f"Failed to start async query: {e}")
            raise ServiceError(f"Failed to start async processing: {str(e)}")
    
    async def get_query_status(self, operation_id: str) -> ProcessingStatus:
        """
        Get status of asynchronous query processing.
        
        Args:
            operation_id: Operation identifier
            
        Returns:
            Current processing status
        """
        if operation_id not in self.active_operations:
            raise ValidationError(f"Operation {operation_id} not found")
        
        return self.active_operations[operation_id]
    
    async def batch_query(self, batch_request: BatchRAGRequest) -> BatchRAGResponse:
        """
        Process multiple queries in batch.
        
        Args:
            batch_request: Batch processing request
            
        Returns:
            Batch processing results
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting batch processing: {len(batch_request.queries)} queries")
            
            successful_responses = []
            failed_queries = []
            
            # Always process in parallel (simplified)
            semaphore = asyncio.Semaphore(batch_request.max_concurrent)
            tasks = []
            
            for query in batch_request.queries:
                task = asyncio.create_task(
                    self._process_batch_query(query, semaphore)
                )
                tasks.append(task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Separate successful and failed results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_queries.append({
                        'query_id': batch_request.queries[i].query_id,
                        'error': str(result)
                    })
                    
                    # Stop on first error if fail_fast is enabled
                    if batch_request.fail_fast:
                        self.logger.warning(f"Batch processing stopped due to fail_fast: {str(result)}")
                        break
                else:
                    successful_responses.append(result)
            
            # Calculate batch statistics
            total_queries = len(batch_request.queries)
            success_count = len(successful_responses)
            success_rate = success_count / total_queries if total_queries > 0 else 0.0
            
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            avg_time = total_time / total_queries if total_queries > 0 else 0.0
            
            # Calculate average confidence (optional)
            avg_confidence = None
            if successful_responses:
                avg_confidence = sum(
                    r.confidence_metrics.overall_confidence 
                    for r in successful_responses
                ) / len(successful_responses)
            
            batch_response = BatchRAGResponse(
                batch_id=batch_request.batch_id,
                total_queries=total_queries,
                successful_responses=successful_responses,
                failed_queries=failed_queries,
                success_rate=success_rate,
                avg_processing_time_ms=avg_time,
                total_processing_time_ms=total_time,
                avg_confidence=avg_confidence
            )
            
            self.logger.info(
                f"Batch processing completed: {success_count}/{total_queries} successful "
                f"({total_time:.2f}ms total, {avg_time:.2f}ms avg)"
            )
            
            return batch_response
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            raise ServiceError(f"Batch processing failed: {str(e)}")
    
    async def add_document(
        self, 
        content: str, 
        title: str, 
        document_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add document to the knowledge base.
        
        Args:
            content: Document content
            title: Document title
            document_type: Type of document
            metadata: Additional metadata
            
        Returns:
            Document processing result
        """
        try:
            # Add document to TextRAG
            document = await self.text_rag.add_document_from_content(
                content=content,
                title=title,
                document_type=document_type,
                metadata=metadata or {}
            )
            
            self.logger.info(f"Document added successfully: {document.document_id}")
            
            return {
                'document_id': document.document_id,
                'title': document.title,
                'status': document.status.value,
                'chunks_created': len(document.chunks) if hasattr(document, 'chunks') else 0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to add document: {e}")
            raise ServiceError(f"Failed to add document: {str(e)}")
    
    async def process_diagram(
        self, 
        image_path: str, 
        diagram_id: str,
        title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process diagram through the complete pipeline.
        
        Args:
            image_path: Path to diagram image
            diagram_id: Unique diagram identifier
            title: Optional diagram title
            
        Returns:
            Processing result
        """
        try:
            # Process diagram through GraphRAG
            result = await self.graph_rag.process_diagram(
                image_path=image_path,
                diagram_id=diagram_id
            )
            
            self.logger.info(f"Diagram processed successfully: {diagram_id}")
            
            return {
                'diagram_id': diagram_id,
                'nodes_created': result.get('nodes_created', 0),
                'relationships_created': result.get('relationships_created', 0),
                'processing_time_ms': result.get('processing_time_ms', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process diagram: {e}")
            raise ServiceError(f"Failed to process diagram: {str(e)}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status and health metrics."""
        try:
            # Get component status
            text_rag_status = "healthy"  # Would check actual service health
            graph_rag_status = "healthy"
            context_manager_status = "healthy"
            
            # Get reliability report
            reliability_report = self.reliability_manager.get_system_reliability_report()
            
            # Active operations
            active_count = len([
                op for op in self.active_operations.values() 
                if op.status not in ['completed', 'failed']
            ])
            
            return {
                'status': 'healthy',
                'services': {
                    'text_rag': text_rag_status,
                    'graph_rag': graph_rag_status,
                    'context_manager': context_manager_status,
                    'orchestrator': 'healthy'
                },
                'active_operations': active_count,
                'performance_stats': self.performance_stats,
                'reliability_report': reliability_report,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'last_updated': datetime.now().isoformat()
            }
    
    # Private methods
    
    def _normalize_query_input(
        self, 
        query: Union[RAGQuery, str, Dict[str, Any]], 
        **kwargs
    ) -> RAGQuery:
        """Normalize different query input formats to RAGQuery object."""
        if isinstance(query, RAGQuery):
            return query
        elif isinstance(query, str):
            # Create RAGQuery from string
            return RAGQuery(
                query_text=query,
                **kwargs
            )
        elif isinstance(query, dict):
            # Create RAGQuery from dict
            query_data = {**query, **kwargs}
            return RAGQuery(**query_data)
        else:
            raise ValidationError(f"Unsupported query type: {type(query)}")
    
    def _validate_query(self, query: RAGQuery) -> None:
        """Validate RAG query parameters."""
        if not query.query_text or not query.query_text.strip():
            raise ValidationError("Query text cannot be empty")
        
        if query.top_k < 1 or query.top_k > 50:
            raise ValidationError("top_k must be between 1 and 50")
        
        if query.similarity_threshold < 0.0 or query.similarity_threshold > 1.0:
            raise ValidationError("similarity_threshold must be between 0.0 and 1.0")
        
        if query.timeout_seconds < 1.0 or query.timeout_seconds > 300.0:
            raise ValidationError("timeout_seconds must be between 1.0 and 300.0")
    
    def _select_workflow(self, query: RAGQuery) -> Any:
        """Select appropriate workflow based on query type."""
        if query.query_type == QueryType.HYBRID_FUSION:
            return self.hybrid_workflow
        elif query.query_type == QueryType.SEMANTIC_SEARCH:
            return self.hybrid_workflow  # Still use hybrid but with text focus
        elif query.query_type == QueryType.GRAPH_TRAVERSAL:
            return self.hybrid_workflow  # Still use hybrid but with graph focus
        else:
            return self.hybrid_workflow  # Default to hybrid workflow
    
    async def _process_async_query(self, operation_id: str, query: RAGQuery) -> None:
        """Process asynchronous query in background."""
        try:
            status = self.active_operations[operation_id]
            status.status = "processing"
            status.current_stage = "executing_workflow"
            
            # Execute query
            response = await self.query(query)
            
            # Update status with result
            status.status = "completed"
            status.progress = 100.0
            status.result = response
            
        except Exception as e:
            # Update status with error
            status = self.active_operations[operation_id]
            status.status = "failed"
            status.error = str(e)
            
            self.logger.error(f"Async query {operation_id} failed: {e}")
    
    async def _process_batch_query(
        self, 
        query: RAGQuery, 
        semaphore: asyncio.Semaphore
    ) -> RAGResponse:
        """Process single query in batch with concurrency control."""
        async with semaphore:
            return await self.query(query)
    
    def _update_performance_stats(self, processing_time: float, success: bool) -> None:
        """Update performance statistics."""
        self.performance_stats['total_queries'] += 1
        self.performance_stats['total_processing_time_ms'] += processing_time
        
        if success:
            self.performance_stats['successful_queries'] += 1
        else:
            self.performance_stats['failed_queries'] += 1
        
        # Update average processing time
        total_queries = self.performance_stats['total_queries']
        total_time = self.performance_stats['total_processing_time_ms']
        self.performance_stats['avg_processing_time_ms'] = total_time / total_queries
    
    async def _create_error_response(self, query_id: str, error_message: str) -> RAGResponse:
        """Create error response for failed queries."""
        from .reliability import ConfidenceCalculator
        
        fallback_metrics = ConfidenceCalculator()._create_fallback_metrics(0)
        
        return RAGResponse(
            query_id=query_id,
            response_text=f"I encountered an error processing your query: {error_message}",
            sources=[],
            visualizations=[],
            confidence_metrics=fallback_metrics,
            context_used=RAGContext(),
            processing_mode=ProcessingMode.FAST,
            processing_time_ms=0.0,
            requires_verification=True,
            has_uncertainties=True,
            suggested_follow_ups=[
                "Try rephrasing your query",
                "Check if all required services are running",
                "Contact support if the issue persists"
            ]
        )
    
    def close(self) -> None:
        """Clean up resources."""
        try:
            self.text_rag.close()
            self.graph_rag.close()
            self.context_manager.close()
            
            if self.db_client:
                self.db_client.close()
            
            self.logger.info("RAG Orchestrator closed successfully")
            
        except Exception as e:
            self.logger.error(f"Error closing RAG Orchestrator: {e}")