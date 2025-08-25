"""
Metrics collection and monitoring for NetBot V2.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Union
from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import wraps

from ...config.settings import get_settings


@dataclass
class Metric:
    """Represents a single metric measurement."""
    
    name: str
    value: Union[int, float]
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    
    @property
    def age_seconds(self) -> float:
        """Get age of metric in seconds."""
        return time.time() - self.timestamp


class MetricsCollector:
    """
    Collects and stores performance metrics for NetBot operations.
    
    Provides thread-safe metric collection with time-series storage
    and basic analytics capabilities.
    """
    
    _instance: Optional["MetricsCollector"] = None
    _lock = threading.RLock()
    
    def __new__(cls) -> "MetricsCollector":
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize metrics collector."""
        if self._initialized:
            return
        
        self.settings = get_settings()
        
        # Storage for metrics (time-series data)
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._timers: Dict[str, List[float]] = defaultdict(list)
        
        # Configuration
        self.max_history = self.settings.monitoring_config.get('max_history', 1000)
        self.retention_seconds = self.settings.monitoring_config.get('retention_seconds', 3600)
        
        self._initialized = True
    
    def counter(self, name: str, value: int = 1, tags: Dict[str, str] = None) -> None:
        """
        Increment a counter metric.
        
        Args:
            name: Counter name
            value: Increment value
            tags: Optional tags
        """
        with self._lock:
            self._counters[name] += value
            
            metric = Metric(
                name=name,
                value=self._counters[name],
                timestamp=time.time(),
                tags=tags or {}
            )
            
            self._metrics[name].append(metric)
            self._cleanup_old_metrics(name)
    
    def gauge(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """
        Set a gauge metric value.
        
        Args:
            name: Gauge name
            value: Current value
            tags: Optional tags
        """
        with self._lock:
            self._gauges[name] = value
            
            metric = Metric(
                name=name,
                value=value,
                timestamp=time.time(),
                tags=tags or {}
            )
            
            self._metrics[name].append(metric)
            self._cleanup_old_metrics(name)
    
    def timer(self, name: str, duration_seconds: float, tags: Dict[str, str] = None) -> None:
        """
        Record a timing metric.
        
        Args:
            name: Timer name
            duration_seconds: Duration in seconds
            tags: Optional tags
        """
        with self._lock:
            self._timers[name].append(duration_seconds)
            
            # Keep only recent timings
            if len(self._timers[name]) > self.max_history:
                self._timers[name] = self._timers[name][-self.max_history:]
            
            metric = Metric(
                name=name,
                value=duration_seconds,
                timestamp=time.time(),
                tags=tags or {}
            )
            
            self._metrics[name].append(metric)
            self._cleanup_old_metrics(name)
    
    def _cleanup_old_metrics(self, name: str) -> None:
        """Remove old metrics beyond retention period."""
        current_time = time.time()
        metrics_queue = self._metrics[name]
        
        # Remove old metrics
        while metrics_queue and current_time - metrics_queue[0].timestamp > self.retention_seconds:
            metrics_queue.popleft()
    
    def get_counter(self, name: str) -> int:
        """Get current counter value."""
        return self._counters.get(name, 0)
    
    def get_gauge(self, name: str) -> float:
        """Get current gauge value.""" 
        return self._gauges.get(name, 0.0)
    
    def get_timer_stats(self, name: str) -> Dict[str, float]:
        """Get timer statistics."""
        timings = self._timers.get(name, [])
        
        if not timings:
            return {
                'count': 0,
                'mean': 0.0,
                'min': 0.0,
                'max': 0.0,
                'p95': 0.0,
                'p99': 0.0
            }
        
        sorted_timings = sorted(timings)
        count = len(sorted_timings)
        
        return {
            'count': count,
            'mean': sum(sorted_timings) / count,
            'min': sorted_timings[0],
            'max': sorted_timings[-1],
            'p95': sorted_timings[int(0.95 * count)] if count > 0 else 0.0,
            'p99': sorted_timings[int(0.99 * count)] if count > 0 else 0.0,
        }
    
    def get_metric_history(self, name: str, limit: int = 100) -> List[Metric]:
        """
        Get recent history for a metric.
        
        Args:
            name: Metric name
            limit: Maximum number of points
            
        Returns:
            List of recent metrics
        """
        with self._lock:
            metrics = list(self._metrics.get(name, []))
            return metrics[-limit:] if limit else metrics
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metric values."""
        with self._lock:
            return {
                'counters': dict(self._counters),
                'gauges': dict(self._gauges),
                'timers': {name: self.get_timer_stats(name) for name in self._timers},
            }
    
    def record_api_request(self, 
                          endpoint: str, 
                          method: str,
                          duration_seconds: float,
                          status_code: int) -> None:
        """Record API request metrics."""
        tags = {
            'endpoint': endpoint,
            'method': method,
            'status_code': str(status_code)
        }
        
        self.counter('api_requests_total', tags=tags)
        self.timer('api_request_duration', duration_seconds, tags=tags)
    
    def record_db_query(self, 
                       query_type: str,
                       duration_seconds: float,
                       success: bool = True) -> None:
        """Record database query metrics."""
        tags = {
            'query_type': query_type,
            'success': str(success)
        }
        
        self.counter('db_queries_total', tags=tags)
        self.timer('db_query_duration', duration_seconds, tags=tags)
    
    def record_embedding_operation(self,
                                 operation: str,
                                 texts_count: int,
                                 duration_seconds: float) -> None:
        """Record embedding operation metrics."""
        tags = {'operation': operation}
        
        self.counter('embedding_operations_total', tags=tags)
        self.gauge('embedding_texts_processed', texts_count, tags=tags)
        self.timer('embedding_operation_duration', duration_seconds, tags=tags)
    
    def record_model_inference(self,
                             model_name: str,
                             duration_seconds: float,
                             tokens_generated: int = 0) -> None:
        """Record model inference metrics."""
        tags = {'model': model_name}
        
        self.counter('model_inferences_total', tags=tags)
        self.timer('model_inference_duration', duration_seconds, tags=tags)
        
        if tokens_generated > 0:
            self.gauge('model_tokens_generated', tokens_generated, tags=tags)


def timed_operation(metric_name: str, tags: Dict[str, str] = None):
    """
    Decorator for timing operations.
    
    Args:
        metric_name: Name of the timing metric
        tags: Optional tags for the metric
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            metrics = get_metrics()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                metrics.timer(metric_name, duration, tags)
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                error_tags = (tags or {}).copy()
                error_tags['error'] = type(e).__name__
                metrics.timer(f"{metric_name}_error", duration, error_tags)
                raise
        
        return wrapper
    return decorator


def counted_operation(metric_name: str, tags: Dict[str, str] = None):
    """
    Decorator for counting operations.
    
    Args:
        metric_name: Name of the counter metric
        tags: Optional tags for the metric
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            metrics = get_metrics()
            
            try:
                result = func(*args, **kwargs)
                metrics.counter(metric_name, tags=tags)
                return result
                
            except Exception as e:
                error_tags = (tags or {}).copy()
                error_tags['error'] = type(e).__name__
                metrics.counter(f"{metric_name}_error", tags=error_tags)
                raise
        
        return wrapper
    return decorator


# Global instance
_metrics_collector = None
_metrics_lock = threading.Lock()


def get_metrics() -> MetricsCollector:
    """
    Get the global metrics collector instance.
    
    Returns:
        MetricsCollector singleton instance
    """
    global _metrics_collector
    
    if _metrics_collector is None:
        with _metrics_lock:
            if _metrics_collector is None:
                _metrics_collector = MetricsCollector()
    
    return _metrics_collector