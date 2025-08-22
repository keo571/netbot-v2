"""
ChromaDB implementation for vector storage in hybrid RAG system.
"""

import json
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from .base_store import BaseVectorStore
from ..chunking.models import DiagramChunk, ChunkType


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB implementation for storing diagram chunks.
    
    Uses ChromaDB as the vector database backend for efficient
    similarity search of diagram chunks.
    """
    
    def __init__(self, 
                 collection_name: str = "diagram_chunks",
                 persist_directory: str = "data/vector_store",
                 embedding_function = None):
        """
        Initialize ChromaDB vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_function: Custom embedding function (optional)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Set up embedding function
        if embedding_function is None:
            # Use default sentence transformers
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        else:
            self.embedding_function = embedding_function
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        except ValueError:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Hybrid RAG diagram chunks"}
            )
    
    def add_chunks(self, chunks: List[DiagramChunk]) -> bool:
        """Add chunks to ChromaDB collection."""
        try:
            if not chunks:
                return True
            
            # Prepare data for ChromaDB
            ids = []
            documents = []
            embeddings = []
            metadatas = []
            
            for chunk in chunks:
                ids.append(chunk.chunk_id)
                documents.append(chunk.get_full_content())
                metadatas.append(self._prepare_metadata(chunk))
                
                # Use chunk embedding if available, otherwise let ChromaDB compute it
                if chunk.embedding:
                    embeddings.append(chunk.embedding)
            
            # Add to collection
            if embeddings and len(embeddings) == len(chunks):
                # Use provided embeddings
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
            else:
                # Let ChromaDB compute embeddings
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
            
            return True
            
        except Exception as e:
            print(f"Error adding chunks to ChromaDB: {e}")
            return False
    
    def search(self, 
               query_embedding: List[float], 
               top_k: int = 10,
               filters: Optional[Dict[str, Any]] = None) -> List[Tuple[DiagramChunk, float]]:
        """Search for similar chunks in ChromaDB."""
        try:
            # Prepare where clause for filtering
            where_clause = {}
            if filters:
                where_clause.update(filters)
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert results to DiagramChunk objects
            chunks_with_scores = []
            
            if results["ids"] and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    document = results["documents"][0][i]
                    metadata = results["metadatas"][0][i]
                    distance = results["distances"][0][i]
                    
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1.0 - distance
                    
                    # Reconstruct DiagramChunk
                    chunk = self._metadata_to_chunk(chunk_id, document, metadata)
                    chunks_with_scores.append((chunk, similarity_score))
            
            return chunks_with_scores
            
        except Exception as e:
            print(f"Error searching ChromaDB: {e}")
            return []
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[DiagramChunk]:
        """Retrieve a specific chunk by ID."""
        try:
            results = self.collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas"]
            )
            
            if results["ids"] and results["ids"][0]:
                document = results["documents"][0]
                metadata = results["metadatas"][0]
                return self._metadata_to_chunk(chunk_id, document, metadata)
            
            return None
            
        except Exception as e:
            print(f"Error retrieving chunk {chunk_id}: {e}")
            return None
    
    def delete_chunks(self, chunk_ids: List[str]) -> bool:
        """Delete chunks from ChromaDB."""
        try:
            self.collection.delete(ids=chunk_ids)
            return True
        except Exception as e:
            print(f"Error deleting chunks: {e}")
            return False
    
    def list_chunks_by_diagram(self, diagram_id: str) -> List[DiagramChunk]:
        """Get all chunks that reference a specific diagram."""
        try:
            results = self.collection.get(
                where={"diagram_id": diagram_id},
                include=["documents", "metadatas"]
            )
            
            chunks = []
            if results["ids"]:
                for i, chunk_id in enumerate(results["ids"]):
                    document = results["documents"][i]
                    metadata = results["metadatas"][i]
                    chunk = self._metadata_to_chunk(chunk_id, document, metadata)
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            print(f"Error listing chunks for diagram {diagram_id}: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the ChromaDB collection."""
        try:
            # Get total count
            total_chunks = self.collection.count()
            
            # Get diagrams with chunks
            results = self.collection.get(include=["metadatas"])
            diagram_ids = set()
            chunk_types = {}
            
            if results["metadatas"]:
                for metadata in results["metadatas"]:
                    if metadata.get("diagram_id"):
                        diagram_ids.add(metadata["diagram_id"])
                    
                    chunk_type = metadata.get("chunk_type", "unknown")
                    chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            return {
                "total_chunks": total_chunks,
                "unique_diagrams": len(diagram_ids),
                "chunk_types": chunk_types,
                "collection_name": self.collection_name
            }
            
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def clear_collection(self) -> bool:
        """Clear all data from the collection."""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Hybrid RAG diagram chunks"}
            )
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False
    
    def _prepare_metadata(self, chunk: DiagramChunk) -> Dict[str, Any]:
        """Prepare metadata for ChromaDB storage."""
        metadata = {
            "diagram_id": chunk.diagram_id or "",
            "chunk_type": chunk.chunk_type.value,
            "source_document": chunk.source_document or "",
            "has_diagram": chunk.has_diagram_reference(),
            "embedding_model": chunk.embedding_model
        }
        
        # Add optional fields
        if chunk.page_number is not None:
            metadata["page_number"] = chunk.page_number
        
        # Add properties
        if chunk.properties:
            # Flatten properties to avoid nested objects (ChromaDB limitation)
            for key, value in chunk.properties.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[f"prop_{key}"] = value
                else:
                    metadata[f"prop_{key}"] = str(value)
        
        return metadata
    
    def _metadata_to_chunk(self, 
                          chunk_id: str, 
                          document: str, 
                          metadata: Dict[str, Any]) -> DiagramChunk:
        """Convert ChromaDB metadata back to DiagramChunk."""
        # Extract properties
        properties = {}
        for key, value in metadata.items():
            if key.startswith("prop_"):
                prop_key = key[5:]  # Remove "prop_" prefix
                properties[prop_key] = value
        
        # Parse chunk type
        chunk_type = ChunkType.PURE_TEXT
        try:
            chunk_type = ChunkType(metadata.get("chunk_type", "pure_text"))
        except ValueError:
            pass
        
        chunk = DiagramChunk(
            chunk_id=chunk_id,
            diagram_id=metadata.get("diagram_id") or None,
            text_content=document,
            chunk_type=chunk_type,
            source_document=metadata.get("source_document") or None,
            page_number=metadata.get("page_number"),
            embedding_model=metadata.get("embedding_model", "sentence-transformers"),
            properties=properties
        )
        
        return chunk