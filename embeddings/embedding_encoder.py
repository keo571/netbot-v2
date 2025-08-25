"""
Core embedding computation for graph nodes.
No database operations - pure computation only.
"""

import numpy as np
from typing import List
from models.graph_models import GraphNode


class EmbeddingEncoder:
    """Handles embedding computation for graph elements"""
    
    # Class-level cached model instance
    _model_instance = None
    _model_loading = False
    
    def __init__(self):
        """Initialize embedding encoder with cached sentence transformer model"""
        self.embedding_model = self._get_or_load_model()
    
    @classmethod
    def _get_or_load_model(cls):
        """Get cached model or load it once"""
        if cls._model_instance is not None:
            return cls._model_instance
            
        if cls._model_loading:
            # Prevent concurrent loading
            import time
            while cls._model_loading and cls._model_instance is None:
                time.sleep(0.1)
            return cls._model_instance
            
        try:
            cls._model_loading = True
            from sentence_transformers import SentenceTransformer
            # Use a fast, good quality model
            cls._model_instance = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Embedding model loaded: all-MiniLM-L6-v2")
            return cls._model_instance
        except ImportError:
            raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")
        except Exception as e:
            raise RuntimeError("Failed to initialize embedding model") from e
        finally:
            cls._model_loading = False
    
    def batch_compute_embeddings(self, nodes: List[GraphNode]) -> List[np.ndarray]:
        """Compute embeddings for a batch of nodes efficiently"""
        
        # Prepare texts for batch processing using same logic as single node
        texts = []
        for node in nodes:
            text_parts = [node.label]  # Main descriptive name
            
            # Add node type for context
            if node.type:
                text_parts.append(f"Type: {node.type}")
            
            # Add all meaningful properties
            for key, value in node.properties.items():
                if value: # Skip empty values
                    if isinstance(value, str) and len(value.strip()) > 0:
                        text_parts.append(f"{key.replace('_', ' ').title()}: {value}")
            
            texts.append(". ".join(text_parts))
        
        # Batch compute embeddings
        embeddings = self.embedding_model.encode(texts)
        
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a query string to embedding vector"""
        
        return self.embedding_model.encode(query)