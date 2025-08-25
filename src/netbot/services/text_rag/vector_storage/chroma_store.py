"""
ChromaDB vector storage for TextRAG.

Provides production-ready vector storage using ChromaDB
for scalable similarity search and metadata filtering.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from ....shared import get_logger
from .base import VectorStore
from ..models import DocumentChunk, SearchResult


class ChromaVectorStore(VectorStore):
    """
    ChromaDB vector storage implementation.
    
    Provides persistent, scalable vector storage with advanced
    similarity search and filtering capabilities.
    """
    
    def __init__(self, persist_directory: Optional[str] = None):
        self.logger = get_logger(__name__)
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.collection_name = "textrag_chunks"
        self.initialized = False
        
        try:
            import chromadb
            self.chromadb = chromadb
        except ImportError:
            self.logger.error("ChromaDB not installed. Install with: pip install chromadb")
            raise ImportError("ChromaDB required for ChromaVectorStore")
    
    def initialize(self, collection_name: str = "textrag_chunks") -> None:
        """Initialize ChromaDB client and collection."""
        try:
            self.collection_name = collection_name
            
            # Initialize client
            if self.persist_directory:
                self.client = self.chromadb.PersistentClient(path=self.persist_directory)
            else:
                self.client = self.chromadb.EphemeralClient()
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=collection_name)
                self.logger.info(f"Found existing ChromaDB collection: {collection_name}")
            except ValueError:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
                )
                self.logger.info(f"Created new ChromaDB collection: {collection_name}")
            
            self.initialized = True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add_chunk(self, chunk: DocumentChunk) -> bool:
        """Add a single chunk to ChromaDB."""
        if not self.initialized:
            self.initialize()
        
        if chunk.embedding is None:
            self.logger.warning(f"Chunk {chunk.chunk_id} has no embedding")
            return False
        
        try:
            # Prepare metadata
            metadata = {
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "character_count": chunk.character_count,
                "word_count": chunk.word_count,
                "created_at": chunk.created_at.isoformat(),
            }
            
            # Add optional metadata
            if chunk.section_title:
                metadata["section_title"] = chunk.section_title
            if chunk.keywords:
                metadata["keywords"] = ",".join(chunk.keywords[:10])  # Limit keywords
            if chunk.topics:
                metadata["topics"] = ",".join(chunk.topics[:5])  # Limit topics
            if chunk.named_entities:
                metadata["named_entities"] = ",".join(chunk.named_entities[:10])
            
            # Add to collection
            self.collection.add(
                ids=[chunk.chunk_id],
                embeddings=[chunk.embedding.tolist()],
                documents=[chunk.content],
                metadatas=[metadata]
            )
            
            self.logger.debug(f"Added chunk to ChromaDB: {chunk.chunk_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add chunk {chunk.chunk_id} to ChromaDB: {e}")
            return False
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Add multiple chunks to ChromaDB in batch."""
        if not self.initialized:
            self.initialize()
        
        # Filter chunks with embeddings
        valid_chunks = [chunk for chunk in chunks if chunk.embedding is not None]
        
        if not valid_chunks:
            self.logger.warning("No chunks with embeddings to add")
            return 0
        
        try:
            # Prepare batch data
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for chunk in valid_chunks:
                ids.append(chunk.chunk_id)
                embeddings.append(chunk.embedding.tolist())
                documents.append(chunk.content)
                
                metadata = {
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "character_count": chunk.character_count,
                    "word_count": chunk.word_count,
                    "created_at": chunk.created_at.isoformat(),
                }
                
                # Add optional metadata
                if chunk.section_title:
                    metadata["section_title"] = chunk.section_title
                if chunk.keywords:
                    metadata["keywords"] = ",".join(chunk.keywords[:10])
                if chunk.topics:
                    metadata["topics"] = ",".join(chunk.topics[:5])
                if chunk.named_entities:
                    metadata["named_entities"] = ",".join(chunk.named_entities[:10])
                
                metadatas.append(metadata)
            
            # Batch add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            self.logger.info(f"Added {len(valid_chunks)} chunks to ChromaDB")
            return len(valid_chunks)
            
        except Exception as e:
            self.logger.error(f"Failed to add chunks to ChromaDB: {e}")
            return 0
    
    def update_chunk(self, chunk: DocumentChunk) -> bool:
        """Update an existing chunk in ChromaDB."""
        if not self.initialized:
            self.initialize()
        
        # ChromaDB doesn't have direct update, so we delete and re-add
        self.delete_chunk(chunk.chunk_id)
        return self.add_chunk(chunk)
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk from ChromaDB."""
        if not self.initialized:
            self.initialize()
        
        try:
            self.collection.delete(ids=[chunk_id])
            self.logger.debug(f"Deleted chunk from ChromaDB: {chunk_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete chunk {chunk_id} from ChromaDB: {e}")
            return False
    
    def delete_document_chunks(self, document_id: str) -> int:
        """Delete all chunks for a document from ChromaDB."""
        if not self.initialized:
            self.initialize()
        
        try:
            # Query to find all chunks for the document
            results = self.collection.get(
                where={"document_id": document_id}
            )
            
            if not results['ids']:
                return 0
            
            # Delete all found chunks
            self.collection.delete(ids=results['ids'])
            
            deleted_count = len(results['ids'])
            self.logger.info(f"Deleted {deleted_count} chunks for document: {document_id}")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to delete chunks for document {document_id}: {e}")
            return 0
    
    def similarity_search(self, 
                         query_embedding: np.ndarray,
                         top_k: int = 5,
                         similarity_threshold: float = 0.7,
                         filters: Optional[Dict[str, Any]] = None) -> List[Tuple[DocumentChunk, float]]:
        """Perform similarity search using ChromaDB."""
        if not self.initialized:
            self.initialize()
        
        if query_embedding is None:
            return []
        
        try:
            # Build where clause for filters
            where_clause = {}
            if filters:
                if 'document_ids' in filters:
                    where_clause["document_id"] = {"$in": filters['document_ids']}
                # Add other filters as needed
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where_clause if where_clause else None
            )
            
            # Convert results to DocumentChunk objects
            chunks_with_scores = []
            
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    chunk_id = results['ids'][0][i]
                    document = results['documents'][0][i]
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i]
                    
                    # Convert distance to similarity score (0-1)
                    similarity = 1.0 - distance
                    
                    # Check similarity threshold
                    if similarity < similarity_threshold:
                        continue
                    
                    # Create DocumentChunk object
                    chunk = self._metadata_to_chunk(chunk_id, document, metadata)
                    chunks_with_scores.append((chunk, similarity))
            
            return chunks_with_scores
            
        except Exception as e:
            self.logger.error(f"Failed to perform similarity search: {e}")
            return []
    
    def hybrid_search(self, 
                     query_embedding: np.ndarray,
                     query_text: str,
                     top_k: int = 5,
                     vector_weight: float = 0.7,
                     lexical_weight: float = 0.3,
                     filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Perform hybrid search with ChromaDB."""
        if not self.initialized:
            self.initialize()
        
        # Get vector search results
        vector_results = self.similarity_search(
            query_embedding, 
            top_k=top_k*2,  # Get more for fusion
            similarity_threshold=0.0,
            filters=filters
        )
        
        # For lexical search, we need to get documents and do text matching
        # This is a simplified implementation
        try:
            where_clause = {}
            if filters and 'document_ids' in filters:
                where_clause["document_id"] = {"$in": filters['document_ids']}
            
            # Get all documents for lexical search
            all_results = self.collection.get(
                where=where_clause if where_clause else None
            )
            
            # Perform lexical matching
            lexical_scores = {}
            query_terms = query_text.lower().split()
            
            if all_results['ids']:
                for i, doc_id in enumerate(all_results['ids']):
                    content = all_results['documents'][i].lower()
                    
                    # Simple TF scoring
                    score = 0.0
                    for term in query_terms:
                        tf = content.count(term) / len(content.split()) if content else 0
                        score += tf
                    
                    if score > 0:
                        lexical_scores[doc_id] = score
            
            # Combine results
            combined_results = {}
            
            # Add vector results
            for chunk, vector_score in vector_results:
                combined_results[chunk.chunk_id] = {
                    'chunk': chunk,
                    'vector_score': vector_score,
                    'lexical_score': lexical_scores.get(chunk.chunk_id, 0.0)
                }
            
            # Add high-scoring lexical results not in vector results
            for chunk_id, lexical_score in lexical_scores.items():
                if chunk_id not in combined_results and lexical_score > 0.1:  # Threshold
                    # Get chunk data
                    chunk_results = self.collection.get(ids=[chunk_id])
                    if chunk_results['ids']:
                        chunk = self._metadata_to_chunk(
                            chunk_id,
                            chunk_results['documents'][0],
                            chunk_results['metadatas'][0]
                        )
                        combined_results[chunk_id] = {
                            'chunk': chunk,
                            'vector_score': 0.0,
                            'lexical_score': lexical_score
                        }
            
            # Create SearchResult objects with fusion scores
            search_results = []
            for result_data in combined_results.values():
                chunk = result_data['chunk']
                vector_score = result_data['vector_score']
                lexical_score = result_data['lexical_score']
                fusion_score = (vector_weight * vector_score) + (lexical_weight * lexical_score)
                
                search_result = SearchResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    similarity_score=vector_score,
                    lexical_score=lexical_score,
                    fusion_score=fusion_score,
                    section_title=chunk.section_title,
                    keywords=chunk.keywords,
                    topics=chunk.topics,
                    named_entities=chunk.named_entities
                )
                
                search_results.append(search_result)
            
            # Sort by fusion score and return top_k
            search_results.sort(key=lambda x: x.fusion_score, reverse=True)
            
            # Set final ranks
            for i, result in enumerate(search_results[:top_k]):
                result.final_rank = i + 1
            
            return search_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Failed to perform hybrid search: {e}")
            return []
    
    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Retrieve a chunk by ID from ChromaDB."""
        if not self.initialized:
            self.initialize()
        
        try:
            results = self.collection.get(ids=[chunk_id])
            
            if not results['ids']:
                return None
            
            return self._metadata_to_chunk(
                chunk_id,
                results['documents'][0],
                results['metadatas'][0]
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get chunk {chunk_id}: {e}")
            return None
    
    def list_documents(self) -> List[str]:
        """List all document IDs in ChromaDB."""
        if not self.initialized:
            self.initialize()
        
        try:
            results = self.collection.get()
            
            # Extract unique document IDs from metadata
            document_ids = set()
            if results['metadatas']:
                for metadata in results['metadatas']:
                    if 'document_id' in metadata:
                        document_ids.add(metadata['document_id'])
            
            return list(document_ids)
            
        except Exception as e:
            self.logger.error(f"Failed to list documents: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about ChromaDB collection."""
        if not self.initialized:
            self.initialize()
        
        try:
            # Get collection info
            count_result = self.collection.count()
            
            # Get sample to determine embedding dimensions
            sample_results = self.collection.peek(limit=1)
            embedding_dim = None
            if sample_results.get('embeddings') and sample_results['embeddings']:
                embedding_dim = len(sample_results['embeddings'][0])
            
            return {
                'backend': 'chromadb',
                'collection_name': self.collection_name,
                'total_chunks': count_result,
                'embedding_dimensions': embedding_dim,
                'persist_directory': self.persist_directory,
                'client_type': 'persistent' if self.persist_directory else 'ephemeral'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get ChromaDB stats: {e}")
            return {'backend': 'chromadb', 'error': str(e)}
    
    def clear(self) -> bool:
        """Clear all data from ChromaDB collection."""
        if not self.initialized:
            self.initialize()
        
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            self.logger.info("Cleared all data from ChromaDB collection")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear ChromaDB collection: {e}")
            return False
    
    def close(self) -> None:
        """Clean up ChromaDB connections."""
        try:
            # ChromaDB client doesn't need explicit closing
            self.client = None
            self.collection = None
            self.initialized = False
            
            self.logger.info("Closed ChromaDB connections")
            
        except Exception as e:
            self.logger.error(f"Error closing ChromaDB: {e}")
    
    def _metadata_to_chunk(self, chunk_id: str, document: str, metadata: Dict[str, Any]) -> DocumentChunk:
        """Convert ChromaDB metadata back to DocumentChunk object."""
        from datetime import datetime
        
        # Parse keywords, topics, named entities
        keywords = metadata.get('keywords', '').split(',') if metadata.get('keywords') else []
        topics = metadata.get('topics', '').split(',') if metadata.get('topics') else []
        named_entities = metadata.get('named_entities', '').split(',') if metadata.get('named_entities') else []
        
        # Filter empty strings
        keywords = [k.strip() for k in keywords if k.strip()]
        topics = [t.strip() for t in topics if t.strip()]
        named_entities = [ne.strip() for ne in named_entities if ne.strip()]
        
        return DocumentChunk(
            chunk_id=chunk_id,
            document_id=metadata['document_id'],
            content=document,
            chunk_index=metadata.get('chunk_index', 0),
            character_count=metadata.get('character_count', 0),
            word_count=metadata.get('word_count', 0),
            section_title=metadata.get('section_title'),
            keywords=keywords,
            topics=topics,
            named_entities=named_entities,
            created_at=datetime.fromisoformat(metadata.get('created_at', datetime.utcnow().isoformat())),
            embedding=None  # Embedding not returned in query results
        )