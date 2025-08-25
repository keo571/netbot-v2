"""
TextRAG repository for NetBot V2.

Handles all database operations for document storage, chunk management,
and metadata persistence. Integrates with the shared database infrastructure.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta

from ...shared.infrastructure.database import BaseRepository
from ...shared import get_logger
from .models import Document, DocumentChunk, DocumentStatus, SearchResultSet


class TextRAGRepository(BaseRepository):
    """
    Repository for TextRAG data operations.
    
    Manages documents, chunks, and search metadata in Neo4j with
    proper indexing and relationship management.
    """
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)
        self.logger.info("TextRAG Repository initialized")
    
    # Implement required abstract methods from BaseRepository
    def create(self, entity) -> Any:
        """Create an entity (delegates to specific methods)."""
        if hasattr(entity, 'document_id') and hasattr(entity, 'content') and hasattr(entity, 'title'):
            return self.create_document(entity)
        elif hasattr(entity, 'chunk_id') and hasattr(entity, 'document_id'):
            return self.create_chunk(entity)
        else:
            raise ValueError(f"Unknown entity type: {type(entity)}")
    
    def get_by_id(self, entity_id: str) -> Any:
        """Get entity by ID (tries document first, then chunk)."""
        # Try document first
        document = self.get_document(entity_id)
        if document:
            return document
        
        # Try chunk
        chunk = self.get_chunk(entity_id)
        if chunk:
            return chunk
        
        return None
    
    def list_by_diagram(self, diagram_id: str) -> List[Any]:
        """List entities by diagram ID (returns documents)."""
        # For TextRAG, we don't have direct diagram association
        # Return empty list or could be extended to link with diagram references
        return []
    
    def update(self, entity) -> Any:
        """Update an entity (delegates to specific methods)."""
        if hasattr(entity, 'document_id') and hasattr(entity, 'content') and hasattr(entity, 'title'):
            return self.update_document(entity)
        elif hasattr(entity, 'chunk_id'):
            # Chunks are typically recreated rather than updated
            return entity
        else:
            raise ValueError(f"Unknown entity type: {type(entity)}")
    
    # === Document Operations ===
    
    def create_document(self, document: Document) -> Document:
        """
        Create a new document in the database.
        
        Args:
            document: Document to create
            
        Returns:
            Created document with updated metadata
        """
        query = """
        CREATE (d:Document {
            document_id: $document_id,
            title: $title,
            content: $content,
            document_type: $document_type,
            source_path: $source_path,
            source_url: $source_url,
            author: $author,
            status: $status,
            file_size: $file_size,
            character_count: $character_count,
            word_count: $word_count,
            tags: $tags,
            categories: $categories,
            diagram_references: $diagram_references,
            chunking_strategy: $chunking_strategy,
            chunk_size: $chunk_size,
            chunk_overlap: $chunk_overlap,
            processing_errors: $processing_errors,
            quality_score: $quality_score,
            metadata: $metadata,
            created_at: $created_at,
            updated_at: $updated_at
        })
        RETURN d
        """
        
        params = {
            'document_id': document.document_id,
            'title': document.title,
            'content': document.content,
            'document_type': document.document_type.value if hasattr(document.document_type, 'value') else str(document.document_type),
            'source_path': document.source_path,
            'source_url': document.source_url,
            'author': document.author,
            'status': document.status.value if hasattr(document.status, 'value') else str(document.status),
            'file_size': document.file_size,
            'character_count': document.character_count,
            'word_count': document.word_count,
            'tags': document.tags,
            'categories': document.categories,
            'diagram_references': document.diagram_references,
            'chunking_strategy': document.chunking_strategy.value if hasattr(document.chunking_strategy, 'value') else str(document.chunking_strategy),
            'chunk_size': document.chunk_size,
            'chunk_overlap': document.chunk_overlap,
            'processing_errors': document.processing_errors,
            'quality_score': document.quality_score,
            'metadata': document.metadata,
            'created_at': document.created_at.isoformat(),
            'updated_at': document.updated_at.isoformat() if document.updated_at else None
        }
        
        try:
            result = self.execute_query(query, params)
            self.logger.info(f"Created document: {document.document_id}")
            return document
        except Exception as e:
            self.logger.error(f"Failed to create document {document.document_id}: {e}")
            raise
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """
        Retrieve a document by ID.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document if found, None otherwise
        """
        query = """
        MATCH (d:Document {document_id: $document_id})
        RETURN d
        """
        
        try:
            result = self.execute_query(query, {'document_id': document_id})
            
            if not result:
                return None
            
            data = result[0]['d']
            return self._neo4j_to_document(data)
            
        except Exception as e:
            self.logger.error(f"Failed to get document {document_id}: {e}")
            return None
    
    def update_document(self, document: Document) -> Document:
        """
        Update an existing document.
        
        Args:
            document: Document with updated data
            
        Returns:
            Updated document
        """
        document.touch()  # Update timestamp
        
        query = """
        MATCH (d:Document {document_id: $document_id})
        SET d.title = $title,
            d.content = $content,
            d.status = $status,
            d.file_size = $file_size,
            d.character_count = $character_count,
            d.word_count = $word_count,
            d.tags = $tags,
            d.categories = $categories,
            d.diagram_references = $diagram_references,
            d.processing_errors = $processing_errors,
            d.quality_score = $quality_score,
            d.metadata = $metadata,
            d.updated_at = $updated_at
        RETURN d
        """
        
        params = {
            'document_id': document.document_id,
            'title': document.title,
            'content': document.content,
            'status': document.status.value if hasattr(document.status, 'value') else str(document.status),
            'file_size': document.file_size,
            'character_count': document.character_count,
            'word_count': document.word_count,
            'tags': document.tags,
            'categories': document.categories,
            'diagram_references': document.diagram_references,
            'processing_errors': document.processing_errors,
            'quality_score': document.quality_score,
            'metadata': document.metadata,
            'updated_at': document.updated_at.isoformat() if document.updated_at else None
        }
        
        try:
            self.execute_query(query, params)
            self.logger.info(f"Updated document: {document.document_id}")
            return document
        except Exception as e:
            self.logger.error(f"Failed to update document {document.document_id}: {e}")
            raise
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks.
        
        Args:
            document_id: Document to delete
            
        Returns:
            True if deleted successfully
        """
        query = """
        MATCH (d:Document {document_id: $document_id})
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:DocumentChunk)
        DETACH DELETE d, c
        RETURN count(d) as deleted_count
        """
        
        try:
            result = self.execute_query(query, {'document_id': document_id})
            deleted = result[0]['deleted_count'] > 0
            
            if deleted:
                self.logger.info(f"Deleted document: {document_id}")
            
            return deleted
        except Exception as e:
            self.logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    def list_documents(self, 
                      status: Optional[DocumentStatus] = None,
                      categories: Optional[List[str]] = None,
                      limit: int = 100) -> List[Document]:
        """
        List documents with optional filtering.
        
        Args:
            status: Filter by document status
            categories: Filter by categories
            limit: Maximum number of documents
            
        Returns:
            List of matching documents
        """
        query_parts = ["MATCH (d:Document)"]
        params = {'limit': limit}
        
        conditions = []
        if status:
            conditions.append("d.status = $status")
            params['status'] = status.value
        
        if categories:
            conditions.append("ANY(cat IN $categories WHERE cat IN d.categories)")
            params['categories'] = categories
        
        if conditions:
            query_parts.append(f"WHERE {' AND '.join(conditions)}")
        
        query_parts.append("RETURN d ORDER BY d.created_at DESC LIMIT $limit")
        query = "\n".join(query_parts)
        
        try:
            result = self.execute_query(query, params)
            documents = [self._neo4j_to_document(row['d']) for row in result]
            
            self.logger.debug(f"Listed {len(documents)} documents")
            return documents
        except Exception as e:
            self.logger.error(f"Failed to list documents: {e}")
            return []
    
    # === Chunk Operations ===
    
    def create_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """
        Create a document chunk and link it to its document.
        
        Args:
            chunk: Document chunk to create
            
        Returns:
            Created chunk
        """
        query = """
        MATCH (d:Document {document_id: $document_id})
        CREATE (c:DocumentChunk {
            chunk_id: $chunk_id,
            document_id: $document_id,
            content: $content,
            chunk_index: $chunk_index,
            start_char: $start_char,
            end_char: $end_char,
            character_count: $character_count,
            word_count: $word_count,
            sentence_count: $sentence_count,
            embedding_model: $embedding_model,
            previous_chunk_id: $previous_chunk_id,
            next_chunk_id: $next_chunk_id,
            section_title: $section_title,
            keywords: $keywords,
            named_entities: $named_entities,
            topics: $topics,
            coherence_score: $coherence_score,
            relevance_score: $relevance_score,
            metadata: $metadata,
            created_at: $created_at,
            updated_at: $updated_at
        })
        CREATE (d)-[:HAS_CHUNK]->(c)
        RETURN c
        """
        
        params = {
            'document_id': chunk.document_id,
            'chunk_id': chunk.chunk_id,
            'content': chunk.content,
            'chunk_index': chunk.chunk_index,
            'start_char': chunk.start_char,
            'end_char': chunk.end_char,
            'character_count': chunk.character_count,
            'word_count': chunk.word_count,
            'sentence_count': chunk.sentence_count,
            'embedding_model': chunk.embedding_model,
            'previous_chunk_id': chunk.previous_chunk_id,
            'next_chunk_id': chunk.next_chunk_id,
            'section_title': chunk.section_title,
            'keywords': chunk.keywords,
            'named_entities': chunk.named_entities,
            'topics': chunk.topics,
            'coherence_score': chunk.coherence_score,
            'relevance_score': chunk.relevance_score,
            'metadata': chunk.metadata,
            'created_at': chunk.created_at.isoformat(),
            'updated_at': chunk.updated_at.isoformat() if chunk.updated_at else None
        }
        
        try:
            result = self.execute_query(query, params)
            self.logger.debug(f"Created chunk: {chunk.chunk_id}")
            return chunk
        except Exception as e:
            self.logger.error(f"Failed to create chunk {chunk.chunk_id}: {e}")
            raise
    
    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """
        Retrieve a chunk by ID.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            DocumentChunk if found, None otherwise
        """
        query = """
        MATCH (c:DocumentChunk {chunk_id: $chunk_id})
        RETURN c
        """
        
        try:
            result = self.execute_query(query, {'chunk_id': chunk_id})
            
            if not result:
                return None
            
            data = result[0]['c']
            return self._neo4j_to_chunk(data)
            
        except Exception as e:
            self.logger.error(f"Failed to get chunk {chunk_id}: {e}")
            return None
    
    def get_document_chunks(self, document_id: str) -> List[DocumentChunk]:
        """
        Get all chunks for a document, ordered by index.
        
        Args:
            document_id: Document identifier
            
        Returns:
            List of chunks ordered by chunk_index
        """
        query = """
        MATCH (d:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:DocumentChunk)
        RETURN c
        ORDER BY c.chunk_index
        """
        
        try:
            result = self.execute_query(query, {'document_id': document_id})
            chunks = [self._neo4j_to_chunk(row['c']) for row in result]
            
            self.logger.debug(f"Retrieved {len(chunks)} chunks for document {document_id}")
            return chunks
        except Exception as e:
            self.logger.error(f"Failed to get chunks for document {document_id}: {e}")
            return []
    
    def update_chunk_embedding(self, chunk_id: str, embedding: Any, model: str) -> bool:
        """
        Update the embedding for a chunk.
        
        Args:
            chunk_id: Chunk to update
            embedding: Vector embedding (will be serialized)
            model: Model name used for embedding
            
        Returns:
            True if updated successfully
        """
        # Note: Neo4j doesn't natively support vector storage
        # In production, embeddings would be stored in a vector database
        # Here we just update the metadata
        
        query = """
        MATCH (c:DocumentChunk {chunk_id: $chunk_id})
        SET c.embedding_model = $model,
            c.updated_at = $updated_at
        RETURN c
        """
        
        params = {
            'chunk_id': chunk_id,
            'model': model,
            'updated_at': datetime.utcnow().isoformat()
        }
        
        try:
            result = self.execute_query(query, params)
            success = len(result) > 0
            
            if success:
                self.logger.debug(f"Updated embedding for chunk: {chunk_id}")
            
            return success
        except Exception as e:
            self.logger.error(f"Failed to update embedding for chunk {chunk_id}: {e}")
            return False
    
    def delete_document_chunks(self, document_id: str) -> int:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: Document whose chunks to delete
            
        Returns:
            Number of chunks deleted
        """
        query = """
        MATCH (d:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:DocumentChunk)
        DETACH DELETE c
        RETURN count(c) as deleted_count
        """
        
        try:
            result = self.execute_query(query, {'document_id': document_id})
            deleted_count = result[0]['deleted_count']
            
            self.logger.info(f"Deleted {deleted_count} chunks for document {document_id}")
            return deleted_count
        except Exception as e:
            self.logger.error(f"Failed to delete chunks for document {document_id}: {e}")
            return 0
    
    # === Search Metadata Operations ===
    
    def record_search(self, search_result_set: SearchResultSet) -> None:
        """
        Record search operation for analytics.
        
        Args:
            search_result_set: Complete search result information
        """
        query = """
        CREATE (s:SearchOperation {
            query_id: $query_id,
            query_text: $query_text,
            search_method: $search_method,
            total_results: $total_results,
            returned_results: $returned_results,
            search_time_ms: $search_time_ms,
            documents_searched: $documents_searched,
            chunks_searched: $chunks_searched,
            filters_applied: $filters_applied,
            session_id: $session_id,
            user_id: $user_id,
            timestamp: $timestamp
        })
        """
        
        params = {
            'query_id': search_result_set.query_id,
            'query_text': search_result_set.query_text,
            'search_method': search_result_set.search_method.value,
            'total_results': search_result_set.total_results,
            'returned_results': search_result_set.returned_results,
            'search_time_ms': search_result_set.search_time_ms,
            'documents_searched': search_result_set.documents_searched,
            'chunks_searched': search_result_set.chunks_searched,
            'filters_applied': search_result_set.filters_applied,
            'session_id': search_result_set.session_id,
            'user_id': search_result_set.user_id,
            'timestamp': search_result_set.timestamp.isoformat()
        }
        
        try:
            self.execute_query(query, params)
            self.logger.debug(f"Recorded search operation: {search_result_set.query_id}")
        except Exception as e:
            self.logger.error(f"Failed to record search operation: {e}")
    
    # === Analytics Operations ===
    
    def get_document_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about documents in the system.
        
        Returns:
            Dictionary with document statistics
        """
        query = """
        MATCH (d:Document)
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:DocumentChunk)
        RETURN 
            count(DISTINCT d) as total_documents,
            count(DISTINCT c) as total_chunks,
            avg(d.character_count) as avg_document_size,
            avg(d.word_count) as avg_word_count,
            collect(DISTINCT d.status) as statuses,
            collect(DISTINCT d.document_type) as types
        """
        
        try:
            result = self.execute_query(query)
            
            if result:
                data = result[0]
                return {
                    'total_documents': data['total_documents'],
                    'total_chunks': data['total_chunks'],
                    'avg_document_size': data['avg_document_size'],
                    'avg_word_count': data['avg_word_count'],
                    'document_statuses': data['statuses'],
                    'document_types': data['types']
                }
            
            return {}
        except Exception as e:
            self.logger.error(f"Failed to get document statistics: {e}")
            return {}
    
    def get_search_statistics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get search statistics for the last N days.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with search statistics
        """
        cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        query = """
        MATCH (s:SearchOperation)
        WHERE s.timestamp >= $cutoff_date
        RETURN 
            count(s) as total_searches,
            avg(s.search_time_ms) as avg_search_time,
            avg(s.returned_results) as avg_results_returned,
            collect(DISTINCT s.search_method) as search_methods
        """
        
        try:
            result = self.execute_query(query, {'cutoff_date': cutoff_date})
            
            if result:
                data = result[0]
                return {
                    'total_searches': data['total_searches'],
                    'avg_search_time_ms': data['avg_search_time'],
                    'avg_results_returned': data['avg_results_returned'],
                    'search_methods_used': data['search_methods']
                }
            
            return {}
        except Exception as e:
            self.logger.error(f"Failed to get search statistics: {e}")
            return {}
    
    # === Helper Methods ===
    
    def _neo4j_to_document(self, data: Dict[str, Any]) -> Document:
        """Convert Neo4j result to Document model."""
        from .models import DocumentType, DocumentStatus, ChunkingStrategy
        
        return Document(
            document_id=data['document_id'],
            title=data['title'],
            content=data['content'],
            document_type=DocumentType(data['document_type']),
            source_path=data.get('source_path'),
            source_url=data.get('source_url'),
            author=data.get('author'),
            status=DocumentStatus(data['status']),
            file_size=data.get('file_size', 0),
            character_count=data.get('character_count', 0),
            word_count=data.get('word_count', 0),
            tags=data.get('tags', []),
            categories=data.get('categories', []),
            diagram_references=data.get('diagram_references', []),
            chunking_strategy=ChunkingStrategy(data.get('chunking_strategy', 'recursive_character')),
            chunk_size=data.get('chunk_size', 1000),
            chunk_overlap=data.get('chunk_overlap', 200),
            processing_errors=data.get('processing_errors', []),
            quality_score=data.get('quality_score', 0.0),
            metadata=data.get('metadata', {}),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else None
        )
    
    def _neo4j_to_chunk(self, data: Dict[str, Any]) -> DocumentChunk:
        """Convert Neo4j result to DocumentChunk model."""
        return DocumentChunk(
            chunk_id=data['chunk_id'],
            document_id=data['document_id'],
            content=data['content'],
            chunk_index=data['chunk_index'],
            start_char=data.get('start_char', 0),
            end_char=data.get('end_char', 0),
            character_count=data.get('character_count', 0),
            word_count=data.get('word_count', 0),
            sentence_count=data.get('sentence_count', 0),
            embedding=None,  # Embeddings would be loaded from vector store
            embedding_model=data.get('embedding_model'),
            previous_chunk_id=data.get('previous_chunk_id'),
            next_chunk_id=data.get('next_chunk_id'),
            section_title=data.get('section_title'),
            keywords=data.get('keywords', []),
            named_entities=data.get('named_entities', []),
            topics=data.get('topics', []),
            coherence_score=data.get('coherence_score', 0.0),
            relevance_score=data.get('relevance_score', 0.0),
            metadata=data.get('metadata', {}),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else None
        )