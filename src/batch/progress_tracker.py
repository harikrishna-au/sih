"""
Progress tracking for batch processing operations.

Provides detailed progress monitoring, status reporting, and estimated
completion times for batch processing operations.
"""

import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from collections import deque

from ..models import BatchProcessingStatus


class ProcessingStatus(Enum):
    """Status of batch processing operation."""
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    PROCESSING = "processing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class FileProcessingInfo:
    """Information about individual file processing."""
    file_path: str
    status: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    retry_count: int = 0


@dataclass
class ProcessingMetrics:
    """Processing performance metrics."""
    files_per_second: float = 0.0
    average_file_time: float = 0.0
    estimated_completion: Optional[float] = None
    time_remaining: Optional[float] = None
    throughput_trend: List[float] = field(default_factory=list)


class ProgressTracker:
    """
    Tracks progress of batch processing operations with detailed metrics
    and estimated completion times.
    """
    
    def __init__(self, metrics_window_size: int = 10):
        """
        Initialize progress tracker.
        
        Args:
            metrics_window_size: Size of sliding window for metrics calculation
        """
        self.metrics_window_size = metrics_window_size
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Processing state
        self._status = ProcessingStatus.NOT_STARTED
        self._total_files = 0
        self._processed_files = 0
        self._failed_files = 0
        self._current_file: Optional[str] = None
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        
        # File tracking
        self._file_info: Dict[str, FileProcessingInfo] = {}
        self._processing_queue: List[str] = []
        self._completed_files: List[str] = []
        self._failed_files_list: List[str] = []
        
        # Performance metrics
        self._completion_times = deque(maxlen=metrics_window_size)
        self._throughput_samples = deque(maxlen=metrics_window_size)
        self._last_throughput_time = None
        self._last_processed_count = 0
        
        # Error tracking
        self._errors: List[str] = []
        self._warnings: List[str] = []
    
    def start_batch(self, total_files: int) -> None:
        """
        Start tracking a new batch processing operation.
        
        Args:
            total_files: Total number of files to process
        """
        with self._lock:
            self._status = ProcessingStatus.INITIALIZING
            self._total_files = total_files
            self._processed_files = 0
            self._failed_files = 0
            self._current_file = None
            self._start_time = time.time()
            self._end_time = None
            
            # Reset tracking data
            self._file_info.clear()
            self._processing_queue.clear()
            self._completed_files.clear()
            self._failed_files_list.clear()
            self._completion_times.clear()
            self._throughput_samples.clear()
            self._errors.clear()
            self._warnings.clear()
            
            self._last_throughput_time = self._start_time
            self._last_processed_count = 0
            
            # Transition to processing
            self._status = ProcessingStatus.PROCESSING
    
    def start_file_processing(self, file_path: str) -> None:
        """
        Mark the start of processing for a specific file.
        
        Args:
            file_path: Path of the file being processed
        """
        with self._lock:
            self._current_file = file_path
            
            file_info = FileProcessingInfo(
                file_path=file_path,
                status="processing",
                start_time=time.time()
            )
            self._file_info[file_path] = file_info
            
            if file_path not in self._processing_queue:
                self._processing_queue.append(file_path)
    
    def mark_file_completed(self, file_path: str, success: bool, error_message: str = None) -> None:
        """
        Mark a file as completed (successfully or with error).
        
        Args:
            file_path: Path of the completed file
            success: Whether processing was successful
            error_message: Error message if processing failed
        """
        with self._lock:
            current_time = time.time()
            
            # Update file info
            if file_path in self._file_info:
                file_info = self._file_info[file_path]
                file_info.end_time = current_time
                file_info.status = "completed" if success else "failed"
                
                if file_info.start_time:
                    file_info.processing_time = current_time - file_info.start_time
                    self._completion_times.append(file_info.processing_time)
                
                if not success:
                    file_info.error_message = error_message
                    self._failed_files_list.append(file_path)
                    self._failed_files += 1
                    if error_message:
                        self._errors.append(f"{file_path}: {error_message}")
                else:
                    self._completed_files.append(file_path)
            
            # Update counters
            self._processed_files += 1
            
            # Update throughput metrics
            self._update_throughput_metrics()
            
            # Clear current file if it matches
            if self._current_file == file_path:
                self._current_file = None
    
    def add_warning(self, message: str) -> None:
        """
        Add a warning message.
        
        Args:
            message: Warning message
        """
        with self._lock:
            self._warnings.append(message)
    
    def add_error(self, message: str) -> None:
        """
        Add an error message.
        
        Args:
            message: Error message
        """
        with self._lock:
            self._errors.append(message)
    
    def pause_processing(self) -> None:
        """Pause the processing operation."""
        with self._lock:
            if self._status == ProcessingStatus.PROCESSING:
                self._status = ProcessingStatus.PAUSED
    
    def resume_processing(self) -> None:
        """Resume the processing operation."""
        with self._lock:
            if self._status == ProcessingStatus.PAUSED:
                self._status = ProcessingStatus.PROCESSING
    
    def complete_batch(self, success: bool = True) -> None:
        """
        Mark the batch processing as completed.
        
        Args:
            success: Whether the batch completed successfully
        """
        with self._lock:
            self._end_time = time.time()
            self._status = ProcessingStatus.COMPLETED if success else ProcessingStatus.FAILED
            self._current_file = None
    
    def cancel_batch(self) -> None:
        """Cancel the batch processing operation."""
        with self._lock:
            self._end_time = time.time()
            self._status = ProcessingStatus.CANCELLED
            self._current_file = None
    
    def get_status(self) -> BatchProcessingStatus:
        """
        Get current batch processing status.
        
        Returns:
            BatchProcessingStatus with current progress information
        """
        with self._lock:
            current_time = time.time()
            
            # Calculate estimated completion
            estimated_completion = None
            if (self._status == ProcessingStatus.PROCESSING and 
                self._processed_files > 0 and 
                self._start_time):
                
                elapsed_time = current_time - self._start_time
                avg_time_per_file = elapsed_time / self._processed_files
                remaining_files = self._total_files - self._processed_files
                
                if remaining_files > 0:
                    estimated_completion = current_time + (avg_time_per_file * remaining_files)
            
            return BatchProcessingStatus(
                total_files=self._total_files,
                processed_files=self._processed_files,
                failed_files=self._failed_files,
                current_file=self._current_file,
                start_time=self._start_time,
                estimated_completion=estimated_completion,
                errors=self._errors.copy()
            )
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """
        Get detailed status information including metrics and file details.
        
        Returns:
            Dictionary with comprehensive status information
        """
        with self._lock:
            current_time = time.time()
            
            # Calculate basic metrics
            elapsed_time = 0.0
            if self._start_time:
                if self._end_time:
                    elapsed_time = self._end_time - self._start_time
                else:
                    elapsed_time = current_time - self._start_time
            
            # Calculate performance metrics
            metrics = self._calculate_metrics()
            
            # Get file details
            file_details = []
            for file_path, info in self._file_info.items():
                file_details.append({
                    'file_path': info.file_path,
                    'status': info.status,
                    'processing_time': info.processing_time,
                    'error_message': info.error_message,
                    'retry_count': info.retry_count
                })
            
            return {
                'status': self._status.value,
                'total_files': self._total_files,
                'processed_files': self._processed_files,
                'failed_files': self._failed_files,
                'success_rate': self._processed_files / max(1, self._total_files),
                'current_file': self._current_file,
                'elapsed_time': elapsed_time,
                'estimated_completion': metrics.estimated_completion,
                'time_remaining': metrics.time_remaining,
                'performance_metrics': {
                    'files_per_second': metrics.files_per_second,
                    'average_file_time': metrics.average_file_time,
                    'throughput_trend': list(metrics.throughput_trend)
                },
                'file_details': file_details,
                'errors': self._errors.copy(),
                'warnings': self._warnings.copy(),
                'completed_files': self._completed_files.copy(),
                'failed_files': self._failed_files_list.copy()
            }
    
    def get_metrics(self) -> ProcessingMetrics:
        """
        Get current processing metrics.
        
        Returns:
            ProcessingMetrics with performance information
        """
        with self._lock:
            return self._calculate_metrics()
    
    def _calculate_metrics(self) -> ProcessingMetrics:
        """Calculate current performance metrics."""
        current_time = time.time()
        
        # Calculate average file processing time
        average_file_time = 0.0
        if self._completion_times:
            average_file_time = sum(self._completion_times) / len(self._completion_times)
        
        # Calculate files per second
        files_per_second = 0.0
        if self._start_time and self._processed_files > 0:
            elapsed_time = current_time - self._start_time
            if elapsed_time > 0:
                files_per_second = self._processed_files / elapsed_time
        
        # Calculate estimated completion and time remaining
        estimated_completion = None
        time_remaining = None
        
        if (self._status == ProcessingStatus.PROCESSING and 
            self._processed_files > 0 and 
            self._start_time):
            
            remaining_files = self._total_files - self._processed_files
            if remaining_files > 0 and average_file_time > 0:
                time_remaining = remaining_files * average_file_time
                estimated_completion = current_time + time_remaining
        
        return ProcessingMetrics(
            files_per_second=files_per_second,
            average_file_time=average_file_time,
            estimated_completion=estimated_completion,
            time_remaining=time_remaining,
            throughput_trend=list(self._throughput_samples)
        )
    
    def _update_throughput_metrics(self) -> None:
        """Update throughput metrics with current sample."""
        current_time = time.time()
        
        if self._last_throughput_time:
            time_delta = current_time - self._last_throughput_time
            
            # Update throughput every 5 seconds or every 10 files
            if (time_delta >= 5.0 or 
                self._processed_files - self._last_processed_count >= 10):
                
                if time_delta > 0:
                    files_delta = self._processed_files - self._last_processed_count
                    throughput = files_delta / time_delta
                    self._throughput_samples.append(throughput)
                
                self._last_throughput_time = current_time
                self._last_processed_count = self._processed_files
    
    def reset(self) -> None:
        """Reset the progress tracker to initial state."""
        with self._lock:
            self._status = ProcessingStatus.NOT_STARTED
            self._total_files = 0
            self._processed_files = 0
            self._failed_files = 0
            self._current_file = None
            self._start_time = None
            self._end_time = None
            
            self._file_info.clear()
            self._processing_queue.clear()
            self._completed_files.clear()
            self._failed_files_list.clear()
            self._completion_times.clear()
            self._throughput_samples.clear()
            self._errors.clear()
            self._warnings.clear()
            
            self._last_throughput_time = None
            self._last_processed_count = 0