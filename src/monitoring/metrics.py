"""Performance monitoring and metrics collection."""

import time
import threading
import psutil
import os
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from contextlib import contextmanager
from enum import Enum

from .logger import get_logger


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricValue:
    """A single metric value with metadata."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class PerformanceStats:
    """Performance statistics for an operation."""
    operation_name: str
    total_calls: int
    total_time: float
    average_time: float
    min_time: float
    max_time: float
    success_count: int
    error_count: int
    success_rate: float
    last_called: datetime


class MetricsCollector:
    """Collects and manages system metrics."""
    
    def __init__(self, max_history_size: int = 10000):
        self.logger = get_logger(__name__)
        self.max_history_size = max_history_size
        self._lock = threading.RLock()
        
        # Metric storage
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._timers: Dict[str, List[float]] = defaultdict(list)
        
        # Metric history
        self._metric_history: deque = deque(maxlen=max_history_size)
        
        # Tags for metrics
        self._metric_tags: Dict[str, Dict[str, str]] = {}
        
        # System metrics
        self._system_metrics_enabled = True
        self._system_metrics_interval = 60  # seconds
        self._last_system_metrics_time = 0
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        with self._lock:
            self._counters[name] += value
            self._record_metric(name, self._counters[name], MetricType.COUNTER, tags)
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric value."""
        with self._lock:
            self._gauges[name] = value
            self._record_metric(name, value, MetricType.GAUGE, tags)
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a value in a histogram."""
        with self._lock:
            self._histograms[name].append(value)
            # Keep only recent values to prevent memory growth
            if len(self._histograms[name]) > 1000:
                self._histograms[name] = self._histograms[name][-1000:]
            self._record_metric(name, value, MetricType.HISTOGRAM, tags)
    
    def record_timer(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record a timing measurement."""
        with self._lock:
            self._timers[name].append(duration)
            # Keep only recent values
            if len(self._timers[name]) > 1000:
                self._timers[name] = self._timers[name][-1000:]
            self._record_metric(name, duration, MetricType.TIMER, tags)
    
    def _record_metric(self, name: str, value: Union[int, float], metric_type: MetricType, tags: Optional[Dict[str, str]] = None):
        """Record a metric in the history."""
        metric = MetricValue(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.now(),
            tags=tags or {},
            unit=self._get_metric_unit(name, metric_type)
        )
        
        self._metric_history.append(metric)
        
        # Store tags for this metric
        if tags:
            self._metric_tags[name] = tags
    
    def _get_metric_unit(self, name: str, metric_type: MetricType) -> str:
        """Get the unit for a metric based on its name and type."""
        if metric_type == MetricType.TIMER or 'time' in name.lower() or 'duration' in name.lower():
            return "seconds"
        elif 'memory' in name.lower() or 'bytes' in name.lower():
            return "bytes"
        elif 'cpu' in name.lower() or 'usage' in name.lower():
            return "percent"
        elif 'count' in name.lower() or metric_type == MetricType.COUNTER:
            return "count"
        else:
            return ""
    
    def get_counter(self, name: str) -> int:
        """Get current counter value."""
        with self._lock:
            return self._counters[name]
    
    def get_gauge(self, name: str) -> Optional[float]:
        """Get current gauge value."""
        with self._lock:
            return self._gauges.get(name)
    
    def get_histogram_stats(self, name: str) -> Optional[Dict[str, float]]:
        """Get histogram statistics."""
        with self._lock:
            values = self._histograms.get(name, [])
            if not values:
                return None
            
            values_sorted = sorted(values)
            n = len(values_sorted)
            
            return {
                'count': n,
                'sum': sum(values_sorted),
                'mean': sum(values_sorted) / n,
                'min': values_sorted[0],
                'max': values_sorted[-1],
                'p50': values_sorted[n // 2],
                'p95': values_sorted[int(n * 0.95)] if n > 0 else 0,
                'p99': values_sorted[int(n * 0.99)] if n > 0 else 0
            }
    
    def get_timer_stats(self, name: str) -> Optional[Dict[str, float]]:
        """Get timer statistics."""
        return self.get_histogram_stats(name)  # Same calculation
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics."""
        with self._lock:
            metrics = {
                'counters': dict(self._counters),
                'gauges': dict(self._gauges),
                'histograms': {name: self.get_histogram_stats(name) for name in self._histograms},
                'timers': {name: self.get_timer_stats(name) for name in self._timers},
                'tags': dict(self._metric_tags)
            }
            
            # Add system metrics if enabled
            if self._system_metrics_enabled:
                metrics['system'] = self._collect_system_metrics()
            
            return metrics
    
    def get_recent_metrics(self, minutes: int = 5) -> List[MetricValue]:
        """Get metrics from the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        with self._lock:
            return [m for m in self._metric_history if m.timestamp >= cutoff_time]
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Process metrics
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info()
            
            return {
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'load_avg': os.getloadavg() if hasattr(os, 'getloadavg') else None
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent,
                    'used': memory.used,
                    'free': memory.free
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': disk.percent
                },
                'process': {
                    'memory_rss': process_memory.rss,
                    'memory_vms': process_memory.vms,
                    'cpu_percent': process.cpu_percent(),
                    'num_threads': process.num_threads(),
                    'open_files': len(process.open_files()),
                    'connections': len(process.connections())
                }
            }
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            return {}
    
    def reset_metrics(self):
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._timers.clear()
            self._metric_history.clear()
            self._metric_tags.clear()
            self.logger.info("All metrics reset")
    
    def enable_system_metrics(self, enabled: bool = True, interval: int = 60):
        """Enable or disable system metrics collection."""
        self._system_metrics_enabled = enabled
        self._system_metrics_interval = interval
        self.logger.info(f"System metrics {'enabled' if enabled else 'disabled'}")


class PerformanceMonitor:
    """Monitor performance of operations and functions."""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self.logger = get_logger(__name__)
        self.metrics = metrics_collector or MetricsCollector()
        self._lock = threading.RLock()
        
        # Performance tracking
        self._operation_stats: Dict[str, PerformanceStats] = {}
        self._active_operations: Dict[str, float] = {}  # operation_id -> start_time
    
    @contextmanager
    def monitor_operation(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager to monitor an operation's performance."""
        operation_id = f"{operation_name}_{threading.get_ident()}_{time.time()}"
        start_time = time.time()
        
        try:
            with self._lock:
                self._active_operations[operation_id] = start_time
            
            self.logger.log_operation_start(operation_name, **(tags or {}))
            yield
            
            # Operation succeeded
            duration = time.time() - start_time
            self._record_operation_success(operation_name, duration, tags)
            
        except Exception as e:
            # Operation failed
            duration = time.time() - start_time
            self._record_operation_error(operation_name, duration, e, tags)
            raise
        finally:
            with self._lock:
                self._active_operations.pop(operation_id, None)
    
    def _record_operation_success(self, operation_name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record a successful operation."""
        with self._lock:
            # Update metrics
            self.metrics.increment_counter(f"{operation_name}.calls", tags=tags)
            self.metrics.increment_counter(f"{operation_name}.success", tags=tags)
            self.metrics.record_timer(f"{operation_name}.duration", duration, tags=tags)
            
            # Update performance stats
            if operation_name not in self._operation_stats:
                self._operation_stats[operation_name] = PerformanceStats(
                    operation_name=operation_name,
                    total_calls=0,
                    total_time=0.0,
                    average_time=0.0,
                    min_time=float('inf'),
                    max_time=0.0,
                    success_count=0,
                    error_count=0,
                    success_rate=0.0,
                    last_called=datetime.now()
                )
            
            stats = self._operation_stats[operation_name]
            stats.total_calls += 1
            stats.success_count += 1
            stats.total_time += duration
            stats.average_time = stats.total_time / stats.total_calls
            stats.min_time = min(stats.min_time, duration)
            stats.max_time = max(stats.max_time, duration)
            stats.success_rate = stats.success_count / stats.total_calls
            stats.last_called = datetime.now()
        
        self.logger.log_operation_end(operation_name, success=True, duration=duration, **(tags or {}))
    
    def _record_operation_error(self, operation_name: str, duration: float, error: Exception, tags: Optional[Dict[str, str]] = None):
        """Record a failed operation."""
        with self._lock:
            # Update metrics
            self.metrics.increment_counter(f"{operation_name}.calls", tags=tags)
            self.metrics.increment_counter(f"{operation_name}.errors", tags=tags)
            self.metrics.record_timer(f"{operation_name}.duration", duration, tags=tags)
            
            # Update performance stats
            if operation_name not in self._operation_stats:
                self._operation_stats[operation_name] = PerformanceStats(
                    operation_name=operation_name,
                    total_calls=0,
                    total_time=0.0,
                    average_time=0.0,
                    min_time=float('inf'),
                    max_time=0.0,
                    success_count=0,
                    error_count=0,
                    success_rate=0.0,
                    last_called=datetime.now()
                )
            
            stats = self._operation_stats[operation_name]
            stats.total_calls += 1
            stats.error_count += 1
            stats.total_time += duration
            stats.average_time = stats.total_time / stats.total_calls
            stats.min_time = min(stats.min_time, duration)
            stats.max_time = max(stats.max_time, duration)
            stats.success_rate = stats.success_count / stats.total_calls if stats.total_calls > 0 else 0.0
            stats.last_called = datetime.now()
        
        self.logger.log_operation_end(operation_name, success=False, duration=duration, error=str(error), **(tags or {}))
    
    def monitor_function(self, operation_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """Decorator to monitor function performance."""
        def decorator(func: Callable):
            nonlocal operation_name
            if operation_name is None:
                operation_name = f"{func.__module__}.{func.__name__}"
            
            def wrapper(*args, **kwargs):
                with self.monitor_operation(operation_name, tags):
                    return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def get_operation_stats(self, operation_name: str) -> Optional[PerformanceStats]:
        """Get performance statistics for an operation."""
        with self._lock:
            return self._operation_stats.get(operation_name)
    
    def get_all_operation_stats(self) -> Dict[str, PerformanceStats]:
        """Get performance statistics for all operations."""
        with self._lock:
            return self._operation_stats.copy()
    
    def get_active_operations(self) -> Dict[str, float]:
        """Get currently active operations and their start times."""
        with self._lock:
            current_time = time.time()
            return {
                op_id: current_time - start_time
                for op_id, start_time in self._active_operations.items()
            }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of system performance."""
        with self._lock:
            total_operations = sum(stats.total_calls for stats in self._operation_stats.values())
            total_errors = sum(stats.error_count for stats in self._operation_stats.values())
            
            # Find slowest operations
            slowest_ops = sorted(
                self._operation_stats.values(),
                key=lambda s: s.average_time,
                reverse=True
            )[:5]
            
            # Find most error-prone operations
            error_prone_ops = sorted(
                [s for s in self._operation_stats.values() if s.error_count > 0],
                key=lambda s: s.error_count,
                reverse=True
            )[:5]
            
            return {
                'total_operations': total_operations,
                'total_errors': total_errors,
                'overall_success_rate': (total_operations - total_errors) / total_operations if total_operations > 0 else 0.0,
                'active_operations_count': len(self._active_operations),
                'slowest_operations': [
                    {
                        'name': op.operation_name,
                        'average_time': op.average_time,
                        'total_calls': op.total_calls
                    }
                    for op in slowest_ops
                ],
                'error_prone_operations': [
                    {
                        'name': op.operation_name,
                        'error_count': op.error_count,
                        'error_rate': op.error_count / op.total_calls,
                        'total_calls': op.total_calls
                    }
                    for op in error_prone_ops
                ],
                'metrics_summary': self.metrics.get_all_metrics()
            }
    
    def reset_performance_stats(self):
        """Reset all performance statistics."""
        with self._lock:
            self._operation_stats.clear()
            self._active_operations.clear()
            self.metrics.reset_metrics()
            self.logger.info("Performance statistics reset")


# Global performance monitor instance
_performance_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _performance_monitor


def monitor_operation(operation_name: str, tags: Optional[Dict[str, str]] = None):
    """Context manager to monitor an operation's performance."""
    return _performance_monitor.monitor_operation(operation_name, tags)


def monitor_function(operation_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
    """Decorator to monitor function performance."""
    return _performance_monitor.monitor_function(operation_name, tags)