"""
Resource management for batch processing operations.

Provides memory management, CPU/GPU utilization optimization, and processing
queue management with priority handling for efficient resource usage.
"""

import psutil
import threading
import time
import logging
import gc
import heapq
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Callable
from enum import Enum
from collections import deque
from queue import PriorityQueue, Empty
from concurrent.futures import ThreadPoolExecutor, Future

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..config import SystemConfig


logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of system resources."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"


@dataclass
class ResourceConstraints:
    """Resource usage constraints and limits."""
    max_memory_usage_percent: float = 80.0
    max_cpu_usage_percent: float = 90.0
    max_gpu_memory_percent: float = 85.0
    min_free_disk_gb: float = 1.0
    max_concurrent_files: int = 4
    max_concurrent_embeddings: int = 2
    memory_cleanup_threshold: float = 75.0
    gpu_memory_cleanup_threshold: float = 80.0


@dataclass
class ResourceUsage:
    """Current system resource usage."""
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_free_gb: float
    gpu_memory_percent: float = 0.0
    gpu_memory_available_gb: float = 0.0
    gpu_utilization_percent: float = 0.0


@dataclass
class ProcessingCapacity:
    """Calculated processing capacity based on resources."""
    max_concurrent_files: int
    max_concurrent_embeddings: int
    recommended_batch_size: int
    memory_per_file_mb: float
    can_use_gpu: bool


@dataclass
class ProcessingTask:
    """Represents a processing task with priority and metadata."""
    task_id: str
    priority: int
    task_type: str  # 'file_processing', 'embedding_generation', 'cleanup'
    payload: Dict[str, Any]
    created_at: float
    estimated_duration: float = 0.0
    retry_count: int = 0
    max_retries: int = 3
    
    def __lt__(self, other):
        """Enable priority queue ordering (higher priority first)."""
        return self.priority > other.priority


@dataclass
class QueueStats:
    """Statistics for processing queue management."""
    total_tasks: int
    pending_tasks: int
    active_tasks: int
    completed_tasks: int
    failed_tasks: int
    average_processing_time: float
    queue_wait_time: float


class ResourceManager:
    """
    Manages system resources for optimal batch processing performance.
    
    Monitors CPU, memory, GPU usage and adjusts processing parameters
    to maintain system stability while maximizing throughput.
    """
    
    def __init__(self, config: SystemConfig, constraints: Optional[ResourceConstraints] = None):
        """
        Initialize resource manager.
        
        Args:
            config: System configuration
            constraints: Resource usage constraints
        """
        self.config = config
        self.constraints = constraints or ResourceConstraints()
        
        # Resource monitoring
        self._monitoring_lock = threading.Lock()
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._resource_history = deque(maxlen=60)  # Keep 1 minute of history
        
        # Current resource state
        self._current_usage: Optional[ResourceUsage] = None
        self._last_cleanup_time = 0.0
        
        # Processing optimization
        self._processing_stats = {
            'files_processed': 0,
            'average_memory_per_file': 0.0,
            'average_processing_time': 0.0,
            'gpu_utilization_history': deque(maxlen=20)
        }
        
        # Initialize GPU detection
        self._gpu_available = self._detect_gpu_availability()
        
        logger.info(f"ResourceManager initialized - GPU available: {self._gpu_available}")
    
    def start_monitoring(self) -> None:
        """Start resource monitoring in background thread."""
        with self._monitoring_lock:
            if not self._monitoring_active:
                self._monitoring_active = True
                self._monitoring_thread = threading.Thread(
                    target=self._monitoring_loop,
                    daemon=True
                )
                self._monitoring_thread.start()
                logger.info("Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        with self._monitoring_lock:
            self._monitoring_active = False
            if self._monitoring_thread:
                self._monitoring_thread.join(timeout=5.0)
                self._monitoring_thread = None
                logger.info("Resource monitoring stopped")
    
    def get_current_usage(self) -> ResourceUsage:
        """
        Get current system resource usage.
        
        Returns:
            ResourceUsage with current system metrics
        """
        if self._current_usage is None:
            self._current_usage = self._measure_resource_usage()
        
        return self._current_usage
    
    def get_processing_capacity(self) -> ProcessingCapacity:
        """
        Calculate optimal processing capacity based on current resources.
        
        Returns:
            ProcessingCapacity with recommended processing parameters
        """
        usage = self.get_current_usage()
        
        # Calculate memory-based constraints
        available_memory_gb = usage.memory_available_gb
        estimated_memory_per_file = max(0.1, self._processing_stats['average_memory_per_file'])
        
        # Conservative estimate: each file processing uses ~200MB base + content size
        memory_based_files = max(1, int(available_memory_gb * 1024 * 0.4 / estimated_memory_per_file))
        
        # CPU-based constraints
        cpu_cores = psutil.cpu_count(logical=False) or 4
        cpu_based_files = max(1, min(cpu_cores, int(cpu_cores * 0.8)))
        
        # Apply configured limits
        max_concurrent_files = min(
            memory_based_files,
            cpu_based_files,
            self.constraints.max_concurrent_files
        )
        
        # GPU-based embedding constraints
        max_concurrent_embeddings = self.constraints.max_concurrent_embeddings
        if self._gpu_available and usage.gpu_memory_percent < 70:
            max_concurrent_embeddings = min(max_concurrent_embeddings * 2, 4)
        elif usage.memory_percent > 70:
            max_concurrent_embeddings = max(1, max_concurrent_embeddings // 2)
        
        # Batch size optimization
        if usage.memory_percent < 50:
            recommended_batch_size = 64
        elif usage.memory_percent < 70:
            recommended_batch_size = 32
        else:
            recommended_batch_size = 16
        
        return ProcessingCapacity(
            max_concurrent_files=max_concurrent_files,
            max_concurrent_embeddings=max_concurrent_embeddings,
            recommended_batch_size=recommended_batch_size,
            memory_per_file_mb=estimated_memory_per_file,
            can_use_gpu=self._gpu_available and usage.gpu_memory_percent < 80
        )
    
    def get_optimal_concurrency(self, total_items: int, operation_type: str) -> int:
        """
        Get optimal concurrency level for a specific operation.
        
        Args:
            total_items: Total number of items to process
            operation_type: Type of operation ('file_processing', 'embedding_generation')
            
        Returns:
            Optimal number of concurrent workers
        """
        capacity = self.get_processing_capacity()
        
        if operation_type == 'file_processing':
            base_concurrency = capacity.max_concurrent_files
        elif operation_type == 'embedding_generation':
            base_concurrency = capacity.max_concurrent_embeddings
        else:
            base_concurrency = 2
        
        # Don't exceed total items
        optimal_concurrency = min(base_concurrency, total_items)
        
        # Ensure at least 1
        return max(1, optimal_concurrency)
    
    def should_cleanup_memory(self) -> bool:
        """
        Check if memory cleanup is needed.
        
        Returns:
            True if memory cleanup should be performed
        """
        usage = self.get_current_usage()
        current_time = time.time()
        
        # Check memory threshold
        if usage.memory_percent > self.constraints.memory_cleanup_threshold:
            return True
        
        # Check GPU memory threshold
        if (self._gpu_available and 
            usage.gpu_memory_percent > self.constraints.gpu_memory_cleanup_threshold):
            return True
        
        # Periodic cleanup (every 5 minutes)
        if current_time - self._last_cleanup_time > 300:
            return True
        
        return False
    
    def cleanup_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """
        Perform memory cleanup operations.
        
        Args:
            aggressive: Whether to perform aggressive cleanup
        
        Returns:
            Dictionary with cleanup results
        """
        cleanup_results = {
            'memory_before': 0.0,
            'memory_after': 0.0,
            'gpu_memory_before': 0.0,
            'gpu_memory_after': 0.0,
            'actions_taken': []
        }
        
        # Measure before cleanup
        usage_before = self.get_current_usage()
        cleanup_results['memory_before'] = usage_before.memory_percent
        cleanup_results['gpu_memory_before'] = usage_before.gpu_memory_percent
        
        # Python garbage collection
        collected = gc.collect()
        if collected > 0:
            cleanup_results['actions_taken'].append(f"Collected {collected} objects")
        
        # Aggressive cleanup if requested
        if aggressive:
            # Force garbage collection multiple times
            for _ in range(3):
                collected += gc.collect()
            
            # Clear all caches
            try:
                import functools
                functools.lru_cache.cache_clear = lambda: None
                cleanup_results['actions_taken'].append("Cleared function caches")
            except:
                pass
        
        # GPU memory cleanup
        if self._gpu_available and TORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if aggressive:
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    cleanup_results['actions_taken'].append("Cleared GPU cache")
            except Exception as e:
                logger.warning(f"GPU memory cleanup failed: {str(e)}")
        
        # Update cleanup time
        self._last_cleanup_time = time.time()
        
        # Measure after cleanup
        usage_after = self.get_current_usage()
        cleanup_results['memory_after'] = usage_after.memory_percent
        cleanup_results['gpu_memory_after'] = usage_after.gpu_memory_percent
        
        memory_freed = usage_before.memory_percent - usage_after.memory_percent
        if memory_freed > 0:
            cleanup_results['actions_taken'].append(f"Freed {memory_freed:.1f}% memory")
        
        logger.info(f"Memory cleanup completed: {cleanup_results['actions_taken']}")
        
        return cleanup_results
    
    def check_resource_constraints(self) -> Tuple[bool, List[str]]:
        """
        Check if current resource usage violates constraints.
        
        Returns:
            Tuple of (constraints_met, list_of_violations)
        """
        usage = self.get_current_usage()
        violations = []
        
        if usage.memory_percent > self.constraints.max_memory_usage_percent:
            violations.append(f"Memory usage {usage.memory_percent:.1f}% exceeds limit {self.constraints.max_memory_usage_percent}%")
        
        if usage.cpu_percent > self.constraints.max_cpu_usage_percent:
            violations.append(f"CPU usage {usage.cpu_percent:.1f}% exceeds limit {self.constraints.max_cpu_usage_percent}%")
        
        if usage.disk_free_gb < self.constraints.min_free_disk_gb:
            violations.append(f"Free disk space {usage.disk_free_gb:.1f}GB below minimum {self.constraints.min_free_disk_gb}GB")
        
        if (self._gpu_available and 
            usage.gpu_memory_percent > self.constraints.max_gpu_memory_percent):
            violations.append(f"GPU memory {usage.gpu_memory_percent:.1f}% exceeds limit {self.constraints.max_gpu_memory_percent}%")
        
        return len(violations) == 0, violations
    
    def update_processing_stats(
        self,
        files_processed: int,
        memory_used_mb: float,
        processing_time: float
    ) -> None:
        """
        Update processing statistics for optimization.
        
        Args:
            files_processed: Number of files processed
            memory_used_mb: Memory used in MB
            processing_time: Processing time in seconds
        """
        stats = self._processing_stats
        
        # Update file count
        stats['files_processed'] += files_processed
        
        # Update average memory per file
        if files_processed > 0:
            current_avg = stats['average_memory_per_file']
            new_avg_memory = memory_used_mb / files_processed
            
            # Exponential moving average
            alpha = 0.1
            stats['average_memory_per_file'] = (
                alpha * new_avg_memory + (1 - alpha) * current_avg
            )
        
        # Update average processing time
        if files_processed > 0:
            current_avg_time = stats['average_processing_time']
            new_avg_time = processing_time / files_processed
            
            alpha = 0.1
            stats['average_processing_time'] = (
                alpha * new_avg_time + (1 - alpha) * current_avg_time
            )
    
    def get_resource_recommendations(self) -> List[str]:
        """
        Get recommendations for resource optimization.
        
        Returns:
            List of optimization recommendations
        """
        usage = self.get_current_usage()
        capacity = self.get_processing_capacity()
        recommendations = []
        
        # Memory recommendations
        if usage.memory_percent > 80:
            recommendations.append("Consider reducing batch size or concurrent file processing")
            recommendations.append("Enable memory cleanup or increase cleanup frequency")
        
        # CPU recommendations
        if usage.cpu_percent > 85:
            recommendations.append("Reduce concurrent file processing to lower CPU usage")
        elif usage.cpu_percent < 30:
            recommendations.append("Consider increasing concurrent file processing for better CPU utilization")
        
        # GPU recommendations
        if self._gpu_available:
            if usage.gpu_memory_percent > 80:
                recommendations.append("Reduce embedding batch size to lower GPU memory usage")
            elif usage.gpu_utilization_percent < 30:
                recommendations.append("Consider increasing embedding batch size for better GPU utilization")
        else:
            recommendations.append("Consider installing CUDA and PyTorch for GPU acceleration")
        
        # Disk recommendations
        if usage.disk_free_gb < 5:
            recommendations.append("Low disk space - consider cleaning up temporary files")
        
        return recommendations
    
    def _detect_gpu_availability(self) -> bool:
        """Detect if GPU is available for processing."""
        if not TORCH_AVAILABLE:
            return False
        
        try:
            return torch.cuda.is_available() and torch.cuda.device_count() > 0
        except Exception:
            return False
    
    def _measure_resource_usage(self) -> ResourceUsage:
        """Measure current system resource usage."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / (1024**3)
        
        # Disk space
        disk = psutil.disk_usage('/')
        disk_free_gb = disk.free / (1024**3)
        
        # GPU metrics
        gpu_memory_percent = 0.0
        gpu_memory_available_gb = 0.0
        gpu_utilization_percent = 0.0
        
        if self._gpu_available and TORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    device = torch.cuda.current_device()
                    gpu_memory_total = torch.cuda.get_device_properties(device).total_memory
                    gpu_memory_allocated = torch.cuda.memory_allocated(device)
                    
                    gpu_memory_percent = (gpu_memory_allocated / gpu_memory_total) * 100
                    gpu_memory_available_gb = (gpu_memory_total - gpu_memory_allocated) / (1024**3)
                    
                    # GPU utilization is harder to get, approximate from memory usage
                    gpu_utilization_percent = min(gpu_memory_percent * 1.2, 100.0)
            except Exception as e:
                logger.debug(f"Failed to get GPU metrics: {str(e)}")
        
        return ResourceUsage(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available_gb=memory_available_gb,
            disk_free_gb=disk_free_gb,
            gpu_memory_percent=gpu_memory_percent,
            gpu_memory_available_gb=gpu_memory_available_gb,
            gpu_utilization_percent=gpu_utilization_percent
        )
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                # Measure current usage
                usage = self._measure_resource_usage()
                self._current_usage = usage
                
                # Add to history
                self._resource_history.append({
                    'timestamp': time.time(),
                    'usage': usage
                })
                
                # Check for automatic cleanup
                if self.should_cleanup_memory():
                    self.cleanup_memory()
                
                # Sleep for monitoring interval
                time.sleep(5.0)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {str(e)}")
                time.sleep(10.0)  # Longer sleep on error
    
    def get_resource_history(self, minutes: int = 5) -> List[Dict[str, Any]]:
        """
        Get resource usage history.
        
        Args:
            minutes: Number of minutes of history to return
            
        Returns:
            List of resource usage samples
        """
        cutoff_time = time.time() - (minutes * 60)
        
        return [
            sample for sample in self._resource_history
            if sample['timestamp'] >= cutoff_time
        ]
    
    def optimize_batch_size(self, base_batch_size: int, content_type: str = 'text') -> int:
        """
        Optimize batch size based on current resource usage.
        
        Args:
            base_batch_size: Base batch size to optimize
            content_type: Type of content being processed
            
        Returns:
            Optimized batch size
        """
        usage = self.get_current_usage()
        
        # Memory-based optimization
        memory_factor = 1.0
        if usage.memory_percent > 80:
            memory_factor = 0.5
        elif usage.memory_percent > 60:
            memory_factor = 0.75
        elif usage.memory_percent < 40:
            memory_factor = 1.5
        
        # GPU-based optimization
        gpu_factor = 1.0
        if self._gpu_available and content_type in ['image', 'embedding']:
            if usage.gpu_memory_percent > 80:
                gpu_factor = 0.5
            elif usage.gpu_memory_percent > 60:
                gpu_factor = 0.75
            elif usage.gpu_memory_percent < 40:
                gpu_factor = 1.25
        
        # Content type adjustments
        content_factor = 1.0
        if content_type == 'image':
            content_factor = 0.5  # Images are memory intensive
        elif content_type == 'audio':
            content_factor = 0.75  # Audio processing is moderately intensive
        
        # Calculate optimized batch size
        optimized_size = int(base_batch_size * memory_factor * gpu_factor * content_factor)
        
        # Ensure reasonable bounds
        return max(1, min(optimized_size, base_batch_size * 2))
    
    def get_memory_pressure_level(self) -> str:
        """
        Get current memory pressure level.
        
        Returns:
            Memory pressure level ('low', 'medium', 'high', 'critical')
        """
        usage = self.get_current_usage()
        
        if usage.memory_percent >= 90:
            return 'critical'
        elif usage.memory_percent >= 75:
            return 'high'
        elif usage.memory_percent >= 50:
            return 'medium'
        else:
            return 'low'
    
    def suggest_processing_strategy(self, task_count: int, task_type: str) -> Dict[str, Any]:
        """
        Suggest optimal processing strategy based on current resources.
        
        Args:
            task_count: Number of tasks to process
            task_type: Type of tasks
            
        Returns:
            Dictionary with processing strategy recommendations
        """
        usage = self.get_current_usage()
        capacity = self.get_processing_capacity()
        
        strategy = {
            'recommended_concurrency': capacity.max_concurrent_files,
            'recommended_batch_size': capacity.recommended_batch_size,
            'use_gpu': capacity.can_use_gpu,
            'memory_cleanup_frequency': 'normal',
            'processing_mode': 'normal'
        }
        
        # Adjust based on memory pressure
        memory_pressure = self.get_memory_pressure_level()
        if memory_pressure == 'critical':
            strategy.update({
                'recommended_concurrency': max(1, capacity.max_concurrent_files // 4),
                'recommended_batch_size': max(1, capacity.recommended_batch_size // 4),
                'memory_cleanup_frequency': 'aggressive',
                'processing_mode': 'conservative'
            })
        elif memory_pressure == 'high':
            strategy.update({
                'recommended_concurrency': max(1, capacity.max_concurrent_files // 2),
                'recommended_batch_size': max(1, capacity.recommended_batch_size // 2),
                'memory_cleanup_frequency': 'frequent'
            })
        
        # Adjust based on task count
        if task_count > 1000:
            strategy['processing_mode'] = 'batch_optimized'
            strategy['memory_cleanup_frequency'] = 'frequent'
        elif task_count < 10:
            strategy['processing_mode'] = 'interactive'
        
        # Task type specific adjustments
        if task_type == 'embedding_generation':
            if not capacity.can_use_gpu:
                strategy['recommended_concurrency'] = max(1, strategy['recommended_concurrency'] // 2)
        elif task_type == 'image_processing':
            strategy['recommended_batch_size'] = max(1, strategy['recommended_batch_size'] // 2)
        
        return strategy
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()


class ProcessingQueueManager:
    """
    Advanced processing queue manager with priority handling and resource optimization.
    
    Manages multiple processing queues with different priorities, implements
    resource-aware scheduling, and provides load balancing across workers.
    """
    
    def __init__(
        self,
        resource_manager: ResourceManager,
        max_queue_size: int = 1000,
        worker_timeout: float = 300.0
    ):
        """
        Initialize processing queue manager.
        
        Args:
            resource_manager: Resource manager instance
            max_queue_size: Maximum number of tasks in queue
            worker_timeout: Timeout for worker tasks in seconds
        """
        self.resource_manager = resource_manager
        self.max_queue_size = max_queue_size
        self.worker_timeout = worker_timeout
        
        # Priority queues for different task types
        self._high_priority_queue = PriorityQueue(maxsize=max_queue_size // 3)
        self._normal_priority_queue = PriorityQueue(maxsize=max_queue_size // 3)
        self._low_priority_queue = PriorityQueue(maxsize=max_queue_size // 3)
        
        # Task tracking
        self._active_tasks: Dict[str, ProcessingTask] = {}
        self._completed_tasks: Dict[str, ProcessingTask] = {}
        self._failed_tasks: Dict[str, ProcessingTask] = {}
        self._task_futures: Dict[str, Future] = {}
        
        # Worker management
        self._executor: Optional[ThreadPoolExecutor] = None
        self._worker_count = 0
        self._is_running = False
        self._shutdown_event = threading.Event()
        
        # Statistics
        self._stats = {
            'total_tasks_submitted': 0,
            'total_tasks_completed': 0,
            'total_tasks_failed': 0,
            'total_processing_time': 0.0,
            'average_queue_wait_time': 0.0,
            'peak_queue_size': 0,
            'worker_utilization': 0.0
        }
        
        # Locks
        self._queue_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        
        logger.info("ProcessingQueueManager initialized")
    
    def start(self, initial_workers: Optional[int] = None) -> None:
        """
        Start the queue manager and worker threads.
        
        Args:
            initial_workers: Initial number of worker threads
        """
        if self._is_running:
            logger.warning("Queue manager already running")
            return
        
        # Determine optimal worker count
        if initial_workers is None:
            capacity = self.resource_manager.get_processing_capacity()
            initial_workers = capacity.max_concurrent_files
        
        self._worker_count = initial_workers
        self._executor = ThreadPoolExecutor(
            max_workers=self._worker_count,
            thread_name_prefix="ProcessingWorker"
        )
        
        self._is_running = True
        self._shutdown_event.clear()
        
        # Start resource monitoring
        self.resource_manager.start_monitoring()
        
        logger.info(f"Queue manager started with {self._worker_count} workers")
    
    def stop(self, timeout: float = 30.0) -> None:
        """
        Stop the queue manager and shutdown workers.
        
        Args:
            timeout: Timeout for graceful shutdown
        """
        if not self._is_running:
            return
        
        logger.info("Stopping queue manager...")
        
        self._is_running = False
        self._shutdown_event.set()
        
        # Cancel pending tasks
        self._cancel_pending_tasks()
        
        # Shutdown executor
        if self._executor:
            self._executor.shutdown(wait=True, timeout=timeout)
            self._executor = None
        
        # Stop resource monitoring
        self.resource_manager.stop_monitoring()
        
        logger.info("Queue manager stopped")
    
    def submit_task(
        self,
        task_id: str,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 0,
        estimated_duration: float = 0.0
    ) -> bool:
        """
        Submit a task to the processing queue.
        
        Args:
            task_id: Unique task identifier
            task_type: Type of task ('file_processing', 'embedding_generation', etc.)
            payload: Task payload data
            priority: Task priority (higher values = higher priority)
            estimated_duration: Estimated processing duration in seconds
            
        Returns:
            True if task was successfully queued, False otherwise
        """
        if not self._is_running:
            logger.error("Cannot submit task - queue manager not running")
            return False
        
        # Check if task already exists
        if task_id in self._active_tasks or task_id in self._completed_tasks:
            logger.warning(f"Task {task_id} already exists")
            return False
        
        # Create task
        task = ProcessingTask(
            task_id=task_id,
            priority=priority,
            task_type=task_type,
            payload=payload,
            created_at=time.time(),
            estimated_duration=estimated_duration
        )
        
        # Select appropriate queue based on priority
        try:
            if priority >= 100:  # High priority
                self._high_priority_queue.put_nowait(task)
            elif priority >= 50:  # Normal priority
                self._normal_priority_queue.put_nowait(task)
            else:  # Low priority
                self._low_priority_queue.put_nowait(task)
            
            with self._stats_lock:
                self._stats['total_tasks_submitted'] += 1
                current_queue_size = self.get_queue_size()
                if current_queue_size > self._stats['peak_queue_size']:
                    self._stats['peak_queue_size'] = current_queue_size
            
            # Schedule task execution
            self._schedule_next_task()
            
            logger.debug(f"Task {task_id} queued with priority {priority}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to queue task {task_id}: {str(e)}")
            return False
    
    def get_task_status(self, task_id: str) -> Optional[str]:
        """
        Get the status of a specific task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task status ('pending', 'active', 'completed', 'failed') or None if not found
        """
        if task_id in self._active_tasks:
            return 'active'
        elif task_id in self._completed_tasks:
            return 'completed'
        elif task_id in self._failed_tasks:
            return 'failed'
        else:
            # Check if in any queue
            for queue in [self._high_priority_queue, self._normal_priority_queue, self._low_priority_queue]:
                with queue.mutex:
                    for task in queue.queue:
                        if task.task_id == task_id:
                            return 'pending'
            return None
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending or active task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if task was cancelled, False otherwise
        """
        # Cancel active task
        if task_id in self._task_futures:
            future = self._task_futures[task_id]
            if future.cancel():
                self._cleanup_task(task_id)
                logger.info(f"Task {task_id} cancelled")
                return True
        
        # Remove from queues (more complex for priority queues)
        removed = self._remove_from_queues(task_id)
        if removed:
            logger.info(f"Task {task_id} removed from queue")
            return True
        
        return False
    
    def get_queue_stats(self) -> QueueStats:
        """
        Get current queue statistics.
        
        Returns:
            QueueStats with current metrics
        """
        with self._stats_lock:
            return QueueStats(
                total_tasks=self._stats['total_tasks_submitted'],
                pending_tasks=self.get_queue_size(),
                active_tasks=len(self._active_tasks),
                completed_tasks=len(self._completed_tasks),
                failed_tasks=len(self._failed_tasks),
                average_processing_time=self._stats.get('average_processing_time', 0.0),
                queue_wait_time=self._stats.get('average_queue_wait_time', 0.0)
            )
    
    def get_queue_size(self) -> int:
        """Get total number of pending tasks across all queues."""
        return (
            self._high_priority_queue.qsize() +
            self._normal_priority_queue.qsize() +
            self._low_priority_queue.qsize()
        )
    
    def optimize_worker_count(self) -> None:
        """Dynamically optimize the number of worker threads based on resource usage."""
        if not self._is_running or not self._executor:
            return
        
        capacity = self.resource_manager.get_processing_capacity()
        optimal_workers = capacity.max_concurrent_files
        
        current_workers = self._worker_count
        queue_size = self.get_queue_size()
        
        # Increase workers if queue is growing and resources allow
        if queue_size > current_workers * 2 and optimal_workers > current_workers:
            new_worker_count = min(optimal_workers, current_workers + 2)
            self._resize_worker_pool(new_worker_count)
        
        # Decrease workers if queue is small and we have excess capacity
        elif queue_size < current_workers // 2 and current_workers > 2:
            new_worker_count = max(2, current_workers - 1)
            self._resize_worker_pool(new_worker_count)
    
    def _schedule_next_task(self) -> None:
        """Schedule the next available task for execution."""
        if not self._is_running or not self._executor:
            return
        
        # Check resource constraints
        constraints_met, violations = self.resource_manager.check_resource_constraints()
        if not constraints_met:
            logger.warning(f"Resource constraints violated: {violations}")
            return
        
        # Get next task from priority queues
        task = self._get_next_task()
        if not task:
            return
        
        # Check if we have available workers
        if len(self._active_tasks) >= self._worker_count:
            # Put task back in queue
            self._requeue_task(task)
            return
        
        # Submit task to executor
        try:
            future = self._executor.submit(self._execute_task, task)
            self._task_futures[task.task_id] = future
            self._active_tasks[task.task_id] = task
            
            # Add completion callback
            future.add_done_callback(lambda f: self._task_completed(task.task_id, f))
            
            logger.debug(f"Task {task.task_id} scheduled for execution")
            
        except Exception as e:
            logger.error(f"Failed to schedule task {task.task_id}: {str(e)}")
            self._mark_task_failed(task, str(e))
    
    def _get_next_task(self) -> Optional[ProcessingTask]:
        """Get the next task from priority queues."""
        # Try high priority first
        try:
            return self._high_priority_queue.get_nowait()
        except Empty:
            pass
        
        # Try normal priority
        try:
            return self._normal_priority_queue.get_nowait()
        except Empty:
            pass
        
        # Try low priority
        try:
            return self._low_priority_queue.get_nowait()
        except Empty:
            pass
        
        return None
    
    def _requeue_task(self, task: ProcessingTask) -> None:
        """Put a task back in the appropriate queue."""
        try:
            if task.priority >= 100:
                self._high_priority_queue.put_nowait(task)
            elif task.priority >= 50:
                self._normal_priority_queue.put_nowait(task)
            else:
                self._low_priority_queue.put_nowait(task)
        except Exception as e:
            logger.error(f"Failed to requeue task {task.task_id}: {str(e)}")
    
    def _execute_task(self, task: ProcessingTask) -> Any:
        """Execute a processing task."""
        start_time = time.time()
        
        try:
            logger.debug(f"Executing task {task.task_id} of type {task.task_type}")
            
            # Perform memory cleanup if needed
            if self.resource_manager.should_cleanup_memory():
                self.resource_manager.cleanup_memory()
            
            # Execute task based on type
            result = self._dispatch_task(task)
            
            processing_time = time.time() - start_time
            
            # Update statistics
            with self._stats_lock:
                self._stats['total_processing_time'] += processing_time
                if self._stats['total_tasks_completed'] > 0:
                    self._stats['average_processing_time'] = (
                        self._stats['total_processing_time'] / self._stats['total_tasks_completed']
                    )
            
            logger.debug(f"Task {task.task_id} completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {str(e)}")
            raise
    
    def _dispatch_task(self, task: ProcessingTask) -> Any:
        """Dispatch task to appropriate handler based on task type."""
        task_type = task.task_type
        payload = task.payload
        
        if task_type == 'file_processing':
            return self._handle_file_processing_task(payload)
        elif task_type == 'embedding_generation':
            return self._handle_embedding_generation_task(payload)
        elif task_type == 'cleanup':
            return self._handle_cleanup_task(payload)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _handle_file_processing_task(self, payload: Dict[str, Any]) -> Any:
        """Handle file processing task."""
        # This would integrate with the document processing pipeline
        # For now, return a placeholder result
        file_path = payload.get('file_path')
        processor = payload.get('processor')
        
        if not file_path or not processor:
            raise ValueError("File processing task missing required parameters")
        
        # Perform actual file processing
        try:
            # Use the real document processor
            from ..processors.router import DocumentRouter
            from ..config import ConfigManager
            
            config_manager = ConfigManager()
            config = config_manager.load_config()
            
            document_router = DocumentRouter(config.processing)
            result = document_router.process_document(file_path)
            
            return {
                'file_path': file_path,
                'status': 'processed' if result.success else 'failed',
                'chunks_created': result.chunks_created if result.success else 0,
                'processing_time': result.processing_time,
                'error_message': result.error_message if not result.success else None
            }
            
        except Exception as e:
            logger.error(f"Error in file processing task: {e}")
            return {
                'file_path': file_path,
                'status': 'failed',
                'chunks_created': 0,
                'error_message': str(e)
            }
    
    def _handle_embedding_generation_task(self, payload: Dict[str, Any]) -> Any:
        """Handle embedding generation task."""
        # This would integrate with the embedding generation pipeline
        content = payload.get('content')
        content_type = payload.get('content_type')
        
        if not content:
            raise ValueError("Embedding generation task missing content")
        
        # Perform actual embedding generation
        try:
            # Use the real embedding generator
            from ..embeddings.unified_embedding_generator import UnifiedEmbeddingGenerator
            from ..config import ConfigManager
            
            config_manager = ConfigManager()
            config = config_manager.load_config()
            
            embedding_generator = UnifiedEmbeddingGenerator(config.embedding)
            
            # Generate embedding based on content type
            if content_type == 'text':
                embedding = embedding_generator.generate_text_embedding(content)
            elif content_type == 'image':
                embedding = embedding_generator.generate_image_embedding(content)
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
            
            return {
                'content': content,
                'content_type': content_type,
                'embedding_generated': True,
                'embedding_dimension': len(embedding) if embedding is not None else 0
            }
            
        except Exception as e:
            logger.error(f"Error in embedding generation task: {e}")
            return {
                'content': content,
                'content_type': content_type,
                'embedding_generated': False,
                'error_message': str(e)
            }
        }
    
    def _handle_cleanup_task(self, payload: Dict[str, Any]) -> Any:
        """Handle cleanup task."""
        cleanup_type = payload.get('cleanup_type', 'memory')
        
        if cleanup_type == 'memory':
            return self.resource_manager.cleanup_memory()
        else:
            return {'cleanup_type': cleanup_type, 'status': 'completed'}
    
    def _task_completed(self, task_id: str, future: Future) -> None:
        """Handle task completion."""
        try:
            result = future.result()
            
            # Move task to completed
            if task_id in self._active_tasks:
                task = self._active_tasks.pop(task_id)
                self._completed_tasks[task_id] = task
                
                with self._stats_lock:
                    self._stats['total_tasks_completed'] += 1
                
                logger.debug(f"Task {task_id} completed successfully")
            
        except Exception as e:
            # Move task to failed
            if task_id in self._active_tasks:
                task = self._active_tasks.pop(task_id)
                self._mark_task_failed(task, str(e))
        
        finally:
            self._cleanup_task(task_id)
            
            # Schedule next task
            self._schedule_next_task()
    
    def _mark_task_failed(self, task: ProcessingTask, error_message: str) -> None:
        """Mark a task as failed."""
        task.retry_count += 1
        
        # Retry if under limit
        if task.retry_count <= task.max_retries:
            logger.warning(f"Task {task.task_id} failed, retrying ({task.retry_count}/{task.max_retries})")
            # Reduce priority for retry
            task.priority = max(0, task.priority - 10)
            self._requeue_task(task)
        else:
            logger.error(f"Task {task.task_id} failed permanently: {error_message}")
            self._failed_tasks[task.task_id] = task
            
            with self._stats_lock:
                self._stats['total_tasks_failed'] += 1
    
    def _cleanup_task(self, task_id: str) -> None:
        """Clean up task resources."""
        if task_id in self._task_futures:
            del self._task_futures[task_id]
        
        if task_id in self._active_tasks:
            del self._active_tasks[task_id]
    
    def _cancel_pending_tasks(self) -> None:
        """Cancel all pending tasks."""
        cancelled_count = 0
        
        # Clear all queues
        for queue in [self._high_priority_queue, self._normal_priority_queue, self._low_priority_queue]:
            while not queue.empty():
                try:
                    queue.get_nowait()
                    cancelled_count += 1
                except Empty:
                    break
        
        # Cancel active tasks
        for task_id, future in self._task_futures.items():
            if future.cancel():
                cancelled_count += 1
        
        logger.info(f"Cancelled {cancelled_count} pending tasks")
    
    def _remove_from_queues(self, task_id: str) -> bool:
        """Remove a specific task from all queues."""
        # This is complex for PriorityQueue, so we'll implement a simple approach
        # In a production system, you might want to use a custom queue implementation
        removed = False
        
        for queue in [self._high_priority_queue, self._normal_priority_queue, self._low_priority_queue]:
            temp_tasks = []
            
            # Extract all tasks
            while not queue.empty():
                try:
                    task = queue.get_nowait()
                    if task.task_id != task_id:
                        temp_tasks.append(task)
                    else:
                        removed = True
                except Empty:
                    break
            
            # Put back non-matching tasks
            for task in temp_tasks:
                try:
                    queue.put_nowait(task)
                except:
                    pass
        
        return removed
    
    def _resize_worker_pool(self, new_size: int) -> None:
        """Resize the worker thread pool."""
        if new_size == self._worker_count:
            return
        
        logger.info(f"Resizing worker pool from {self._worker_count} to {new_size}")
        
        # For ThreadPoolExecutor, we need to create a new one
        old_executor = self._executor
        
        self._executor = ThreadPoolExecutor(
            max_workers=new_size,
            thread_name_prefix="ProcessingWorker"
        )
        self._worker_count = new_size
        
        # Shutdown old executor
        if old_executor:
            old_executor.shutdown(wait=False)
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()