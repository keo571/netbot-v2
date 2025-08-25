"""
Advanced cache management for NetBot V2.

Provides flexible caching infrastructure supporting multiple backends
and specialized cache types for different data patterns.
"""

import threading
import time
import pickle
import hashlib
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Callable, List
from functools import wraps
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...config.settings import get_settings
from ...exceptions import CacheError


@dataclass
class CacheEntry:
    """Represents a cached item with metadata."""
    
    value: Any
    created_at: float
    accessed_at: float
    access_count: int = 0
    ttl_seconds: Optional[float] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds
    
    def touch(self):
        """Update access timestamp and count."""
        self.accessed_at = time.time()
        self.access_count += 1


class CacheBackend(ABC):
    """Abstract base for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get cached value by key."""
        pass
    
    @abstractmethod
    def set(self, key: str, entry: CacheEntry) -> None:
        """Set cached value."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete cached value."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cached values."""
        pass
    
    @abstractmethod
    def keys(self) -> List[str]:
        """Get all cache keys."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get cache size."""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache backend with LRU eviction."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get cached value by key."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            
            if entry.is_expired:
                self.delete(key)
                return None
            
            entry.touch()
            # Move to end for LRU
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
            return entry
    
    def set(self, key: str, entry: CacheEntry) -> None:
        """Set cached value."""
        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()
            
            self._cache[key] = entry
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
    
    def delete(self, key: str) -> bool:
        """Delete cached value."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cached values."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
    
    def keys(self) -> List[str]:
        """Get all cache keys."""
        with self._lock:
            return list(self._cache.keys())
    
    def size(self) -> int:
        """Get cache size."""
        return len(self._cache)
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if self._access_order:
            lru_key = self._access_order.pop(0)
            del self._cache[lru_key]


class FileCache(CacheBackend):
    """File-based cache backend for persistent storage."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
    
    def _key_to_path(self, key: str) -> Path:
        """Convert cache key to file path."""
        # Use hash to avoid filesystem issues with key names
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get cached value by key."""
        with self._lock:
            path = self._key_to_path(key)
            if not path.exists():
                return None
            
            try:
                with open(path, 'rb') as f:
                    entry = pickle.load(f)
                
                if entry.is_expired:
                    self.delete(key)
                    return None
                
                entry.touch()
                # Save updated access info
                with open(path, 'wb') as f:
                    pickle.dump(entry, f)
                
                return entry
                
            except Exception as e:
                # Corrupt cache file, delete it
                path.unlink(missing_ok=True)
                return None
    
    def set(self, key: str, entry: CacheEntry) -> None:
        """Set cached value."""
        with self._lock:
            path = self._key_to_path(key)
            try:
                with open(path, 'wb') as f:
                    pickle.dump(entry, f)
            except Exception as e:
                raise CacheError(f"Failed to write cache file: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete cached value."""
        with self._lock:
            path = self._key_to_path(key)
            if path.exists():
                path.unlink()
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cached values."""
        with self._lock:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
    
    def keys(self) -> List[str]:
        """Get all cache keys (expensive operation)."""
        # This is inefficient for file cache, consider keeping an index
        return []
    
    def size(self) -> int:
        """Get cache size."""
        return len(list(self.cache_dir.glob("*.cache")))


class CacheManager:
    """
    Advanced cache manager with multiple backends and specialized caches.
    
    Supports different cache strategies for different data types and use cases.
    """
    
    _instance: Optional["CacheManager"] = None
    _lock = threading.RLock()
    
    def __new__(cls) -> "CacheManager":
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize cache manager."""
        if self._initialized:
            return
        
        self.settings = get_settings()
        
        # Initialize backends
        self.memory_backend = MemoryCache(max_size=self.settings.cache_config.get('memory_size', 1000))
        
        cache_dir = self.settings.cache_config.get('file_cache_dir', 'data/cache')
        self.file_backend = FileCache(cache_dir)
        
        # Cache namespaces for different data types
        self._namespaces = {
            'embeddings': 'mem',  # Fast access needed
            'models': 'mem',      # Keep in memory
            'diagrams': 'file',   # Persistent storage
            'search_results': 'mem',  # Fast but temporary
            'visualizations': 'file',  # Large, persistent
        }
        
        self._initialized = True
    
    def _get_backend(self, namespace: str) -> CacheBackend:
        """Get appropriate backend for namespace."""
        backend_type = self._namespaces.get(namespace, 'mem')
        if backend_type == 'file':
            return self.file_backend
        return self.memory_backend
    
    def _make_key(self, namespace: str, key: str) -> str:
        """Create namespaced key."""
        return f"{namespace}:{key}"
    
    def get(self, key: str, namespace: str = 'default') -> Any:
        """
        Get cached value.
        
        Args:
            key: Cache key
            namespace: Cache namespace
            
        Returns:
            Cached value or None
        """
        backend = self._get_backend(namespace)
        full_key = self._make_key(namespace, key)
        
        entry = backend.get(full_key)
        return entry.value if entry else None
    
    def set(self, 
           key: str, 
           value: Any, 
           namespace: str = 'default',
           ttl_seconds: Optional[float] = None) -> None:
        """
        Set cached value.
        
        Args:
            key: Cache key
            value: Value to cache
            namespace: Cache namespace
            ttl_seconds: Time to live in seconds
        """
        backend = self._get_backend(namespace)
        full_key = self._make_key(namespace, key)
        
        entry = CacheEntry(
            value=value,
            created_at=time.time(),
            accessed_at=time.time(),
            ttl_seconds=ttl_seconds
        )
        
        backend.set(full_key, entry)
    
    def delete(self, key: str, namespace: str = 'default') -> bool:
        """
        Delete cached value.
        
        Args:
            key: Cache key
            namespace: Cache namespace
            
        Returns:
            True if deleted
        """
        backend = self._get_backend(namespace)
        full_key = self._make_key(namespace, key)
        return backend.delete(full_key)
    
    def clear_namespace(self, namespace: str) -> None:
        """Clear all items in a namespace."""
        backend = self._get_backend(namespace)
        prefix = f"{namespace}:"
        
        for key in backend.keys():
            if key.startswith(prefix):
                backend.delete(key)
    
    def clear_all(self) -> None:
        """Clear all caches."""
        self.memory_backend.clear()
        self.file_backend.clear()
    
    def cache_embedding_matrix(self, diagram_id: str, embeddings: np.ndarray) -> None:
        """
        Cache embedding matrix for a diagram.
        
        Args:
            diagram_id: Diagram identifier
            embeddings: Stacked embedding matrix
        """
        key = f"matrix_{diagram_id}"
        self.set(key, embeddings, namespace='embeddings', ttl_seconds=3600)  # 1 hour
    
    def get_embedding_matrix(self, diagram_id: str) -> Optional[np.ndarray]:
        """
        Get cached embedding matrix.
        
        Args:
            diagram_id: Diagram identifier
            
        Returns:
            Embedding matrix or None
        """
        key = f"matrix_{diagram_id}"
        return self.get(key, namespace='embeddings')
    
    def cache_model(self, model_name: str, model: Any) -> None:
        """
        Cache a loaded model.
        
        Args:
            model_name: Model identifier
            model: Model instance
        """
        self.set(model_name, model, namespace='models')
    
    def get_model(self, model_name: str) -> Any:
        """
        Get cached model.
        
        Args:
            model_name: Model identifier
            
        Returns:
            Model instance or None
        """
        return self.get(model_name, namespace='models')
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'memory_backend': {
                'size': self.memory_backend.size(),
                'max_size': self.memory_backend.max_size,
            },
            'file_backend': {
                'size': self.file_backend.size(),
            },
            'namespaces': list(self._namespaces.keys()),
        }


def cache_with_key(
    namespace: str = 'default',
    ttl_seconds: Optional[float] = None,
    key_func: Optional[Callable] = None
):
    """
    Decorator for caching function results.
    
    Args:
        namespace: Cache namespace
        ttl_seconds: Time to live
        key_func: Custom key generation function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache_manager()
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key from function name and arguments
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5('_'.join(key_parts).encode()).hexdigest()
            
            # Try to get from cache
            cached_result = cache.get(cache_key, namespace)
            if cached_result is not None:
                return cached_result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, namespace, ttl_seconds)
            
            return result
        return wrapper
    return decorator


# Global instance
_cache_manager = None
_cache_lock = threading.Lock()


def get_cache_manager() -> CacheManager:
    """
    Get the global cache manager instance.
    
    Returns:
        CacheManager singleton instance
    """
    global _cache_manager
    
    if _cache_manager is None:
        with _cache_lock:
            if _cache_manager is None:
                _cache_manager = CacheManager()
    
    return _cache_manager


# Cleanup function for application shutdown
def close_caches():
    """Close all caches on application shutdown."""
    global _cache_manager
    if _cache_manager:
        _cache_manager.clear_all()