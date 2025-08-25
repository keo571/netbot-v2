"""
Centralized embedding client for vector operations.

Manages embedding models with caching and batch processing capabilities.
"""

import threading
import time
from typing import List, Optional, Dict, Any, Union
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from ...config.settings import get_settings
from ...exceptions import AIError
from ..cache.cache_manager import get_cache_manager


class EmbeddingClient:
    """
    Centralized client for embedding operations.
    
    Provides caching, batch processing, and optimized embedding generation
    for text and other modalities.
    """
    
    _instance: Optional["EmbeddingClient"] = None
    _lock = threading.RLock()
    _initialized = False
    
    def __new__(cls) -> "EmbeddingClient":
        """Singleton pattern for embedding client."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize embedding client."""
        if self._initialized:
            return
        
        self.settings = get_settings()
        self.cache = get_cache_manager()
        self._model_cache: Dict[str, SentenceTransformer] = {}
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._model_loading_locks: Dict[str, threading.Lock] = {}
        
        # Default model configuration
        self.default_model = self.settings.embedding_config.get(
            'default_model', 
            'sentence-transformers/all-MiniLM-L6-v2'
        )
        
        self._initialized = True
    
    def _get_model_lock(self, model_name: str) -> threading.Lock:
        """Get or create a lock for model loading."""
        if model_name not in self._model_loading_locks:
            self._model_loading_locks[model_name] = threading.Lock()
        return self._model_loading_locks[model_name]
    
    def get_model(self, model_name: str = None) -> SentenceTransformer:
        """
        Get or load embedding model with caching.
        
        Args:
            model_name: Model name (uses default if None)
            
        Returns:
            SentenceTransformer model instance
        """
        if model_name is None:
            model_name = self.default_model
        
        # Check cache first
        cached_model = self.cache.get_model(f"embedding_{model_name}")
        if cached_model is not None:
            return cached_model
        
        # Thread-safe model loading
        lock = self._get_model_lock(model_name)
        with lock:
            # Check cache again after acquiring lock
            cached_model = self.cache.get_model(f"embedding_{model_name}")
            if cached_model is not None:
                return cached_model
            
            try:
                print(f"🔄 Loading embedding model: {model_name}")
                start_time = time.time()
                
                model = SentenceTransformer(model_name)
                
                load_time = time.time() - start_time
                print(f"✅ Loaded embedding model in {load_time:.2f}s: {model_name}")
                
                # Cache the model
                self.cache.cache_model(f"embedding_{model_name}", model)
                
                return model
                
            except Exception as e:
                raise AIError(f"Failed to load embedding model {model_name}: {e}")
    
    def encode_text(self,
                   texts: Union[str, List[str]],
                   model_name: str = None,
                   normalize_embeddings: bool = True,
                   batch_size: int = 32) -> np.ndarray:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: Text or list of texts to encode
            model_name: Embedding model to use
            normalize_embeddings: Whether to normalize embeddings
            batch_size: Batch size for processing
            
        Returns:
            Embedding array (single vector or matrix)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        model = self.get_model(model_name)
        
        try:
            embeddings = model.encode(
                texts,
                normalize_embeddings=normalize_embeddings,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True
            )
            
            return embeddings
            
        except Exception as e:
            raise AIError(f"Text encoding failed: {e}")
    
    def encode_single_text(self,
                          text: str,
                          model_name: str = None,
                          use_cache: bool = True) -> np.ndarray:
        """
        Encode single text with caching support.
        
        Args:
            text: Text to encode
            model_name: Model to use
            use_cache: Whether to use/store in cache
            
        Returns:
            Embedding vector
        """
        if model_name is None:
            model_name = self.default_model
        
        # Check cache if enabled
        if use_cache:
            cache_key = f"{model_name}_{hash(text)}"
            cached_embedding = self.cache.get(cache_key, namespace='embeddings')
            if cached_embedding is not None:
                return cached_embedding
        
        # Generate embedding
        embedding = self.encode_text(text, model_name)[0]
        
        # Cache result if enabled
        if use_cache:
            cache_key = f"{model_name}_{hash(text)}"
            self.cache.set(
                cache_key, 
                embedding, 
                namespace='embeddings', 
                ttl_seconds=3600  # 1 hour
            )
        
        return embedding
    
    def compute_similarity(self,
                          embedding1: np.ndarray,
                          embedding2: np.ndarray,
                          metric: str = "cosine") -> float:
        """
        Compute similarity between embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            metric: Similarity metric ("cosine", "dot", "euclidean")
            
        Returns:
            Similarity score
        """
        try:
            if metric == "cosine":
                # Cosine similarity
                dot_product = np.dot(embedding1, embedding2)
                norms = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
                return dot_product / norms if norms > 0 else 0.0
                
            elif metric == "dot":
                # Dot product
                return float(np.dot(embedding1, embedding2))
                
            elif metric == "euclidean":
                # Negative euclidean distance (higher = more similar)
                return -float(np.linalg.norm(embedding1 - embedding2))
                
            else:
                raise ValueError(f"Unknown similarity metric: {metric}")
                
        except Exception as e:
            raise AIError(f"Similarity computation failed: {e}")
    
    def find_most_similar(self,
                         query_embedding: np.ndarray,
                         candidate_embeddings: np.ndarray,
                         top_k: int = 5,
                         metric: str = "cosine") -> List[tuple]:
        """
        Find most similar embeddings from candidates.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: Matrix of candidate embeddings
            top_k: Number of top results to return
            metric: Similarity metric
            
        Returns:
            List of (index, similarity_score) tuples
        """
        try:
            similarities = []
            
            for i, candidate in enumerate(candidate_embeddings):
                similarity = self.compute_similarity(
                    query_embedding, 
                    candidate, 
                    metric
                )
                similarities.append((i, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            raise AIError(f"Similarity search failed: {e}")
    
    def create_embedding_matrix(self,
                               texts: List[str],
                               model_name: str = None,
                               batch_size: int = 32) -> np.ndarray:
        """
        Create embedding matrix from list of texts.
        
        Args:
            texts: List of texts to embed
            model_name: Model to use
            batch_size: Processing batch size
            
        Returns:
            Embedding matrix (num_texts x embedding_dim)
        """
        if not texts:
            return np.array([])
        
        try:
            embeddings = self.encode_text(
                texts,
                model_name=model_name,
                batch_size=batch_size
            )
            
            return embeddings
            
        except Exception as e:
            raise AIError(f"Embedding matrix creation failed: {e}")
    
    def cache_diagram_embeddings(self,
                                diagram_id: str,
                                texts: List[str],
                                model_name: str = None) -> np.ndarray:
        """
        Cache embeddings for a diagram.
        
        Args:
            diagram_id: Diagram identifier
            texts: Node/relationship texts
            model_name: Model to use
            
        Returns:
            Stacked embedding matrix
        """
        try:
            # Check if already cached
            cached_matrix = self.cache.get_embedding_matrix(diagram_id)
            if cached_matrix is not None:
                print(f"✅ Using cached embeddings for diagram: {diagram_id}")
                return cached_matrix
            
            print(f"🔄 Creating embeddings for diagram: {diagram_id}")
            start_time = time.time()
            
            # Create embedding matrix
            embedding_matrix = self.create_embedding_matrix(texts, model_name)
            
            # Cache the matrix
            self.cache.cache_embedding_matrix(diagram_id, embedding_matrix)
            
            elapsed = time.time() - start_time
            print(f"✅ Cached {len(texts)} embeddings in {elapsed:.2f}s for: {diagram_id}")
            
            return embedding_matrix
            
        except Exception as e:
            raise AIError(f"Diagram embedding caching failed: {e}")
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding client statistics."""
        cache_stats = self.cache.get_stats()
        
        return {
            'models_cached': len([k for k in cache_stats.get('memory_backend', {}) if 'embedding_' in str(k)]),
            'embeddings_cached': cache_stats.get('memory_backend', {}).get('size', 0),
            'default_model': self.default_model,
        }


# Global instance
_embedding_client = None
_embedding_lock = threading.Lock()


@lru_cache()
def get_embedding_client() -> EmbeddingClient:
    """
    Get the global embedding client instance.
    
    Returns:
        EmbeddingClient singleton instance
    """
    global _embedding_client
    
    if _embedding_client is None:
        with _embedding_lock:
            if _embedding_client is None:
                _embedding_client = EmbeddingClient()
    
    return _embedding_client