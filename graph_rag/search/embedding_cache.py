"""
Embedding cache for fast vector similarity search.
Caches embeddings and metadata at diagram level to avoid database roundtrips.
"""

import numpy as np
from typing import Dict, List, Tuple


class EmbeddingCache:
    """Caches embeddings and metadata for fast vector search"""
    
    def __init__(self, data_access):
        self.data_access = data_access
        
        # Cache for single diagram
        self._cached_diagram_id: str = None
        self._cached_data: 'CachedDiagram' = None
    
    def get_cached_nodes(self, diagram_id: str) -> Tuple[np.ndarray, List[Dict]]:
        """
        Get cached node embeddings for a diagram.
        Returns: (embedding_matrix, node_data)
        """
        if self._cached_diagram_id != diagram_id:
            self._load_diagram_cache(diagram_id)
        
        return self._cached_data.node_embeddings, self._cached_data.node_data
    
    def _load_diagram_cache(self, diagram_id: str):
        """Load all embeddings and metadata for a diagram into cache"""
        print(f"Loading embeddings cache for diagram: {diagram_id}")
        
        # Load all nodes (embeddings already verified to exist by caller)
        nodes = self.data_access.get_all_nodes(diagram_id)
        
        # Create cached diagram
        cached_diagram = CachedDiagram()
        
        # Cache nodes (all nodes have embeddings since it's all-or-nothing per diagram)
        if nodes:
            cached_diagram.node_embeddings = np.stack([n.embedding for n in nodes])
            cached_diagram.node_data = [
                {
                    'id': n.id,
                    'label': n.label, 
                    'type': n.type,
                    'properties': n.properties
                } for n in nodes
            ]
        else:
            cached_diagram.node_embeddings = np.empty((0, 0))
            cached_diagram.node_data = []
        
        self._cached_diagram_id = diagram_id
        self._cached_data = cached_diagram
        print(f"✅ Cached embedding matrix: {cached_diagram.node_embeddings.shape} for {len(cached_diagram.node_data)} nodes")
    
    def invalidate_cache(self, diagram_id: str):
        """
        Clear cached data for a diagram (call when diagram data changes).
        
        Should be called after:
        - Adding embeddings to a diagram
        - Removing embeddings from a diagram  
        - Re-processing a diagram
        """
        if self._cached_diagram_id == diagram_id:
            self._cached_diagram_id = None
            self._cached_data = None
            print(f"🗑️ Cleared cache for diagram: {diagram_id}")
    
    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics for debugging and monitoring.
        
        Useful for:
        - CLI diagnostics
        - Performance monitoring
        - Understanding cache state
        """
        if self._cached_data:
            return {
                'cached_diagram': self._cached_diagram_id,
                'cached_nodes_with_embeddings': len(self._cached_data.node_data),
                'is_cached': True
            }
        else:
            return {
                'cached_diagram': None,
                'cached_nodes_with_embeddings': 0,
                'is_cached': False
            }


class CachedDiagram:
    """Container for cached diagram data"""
    
    def __init__(self):
        # Node data  
        self.node_embeddings: np.ndarray = None  # Shape: [num_nodes, embedding_dim]
        self.node_data: List[Dict] = []  # Complete node info: [{'id': '...', 'label': '...', 'type': '...', 'properties': {...}}]