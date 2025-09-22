"""
Batch processing coordinator for multimodal RAG system.

Orchestrates multi-file processing with progress tracking, error handling,
and recovery mechanisms for efficient batch ingestion.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union
import threading
from dataclasses import dataclass, field

from ..models import (
    DocumentContent, ProcessingResult, BatchProcessingStatus, 
    ContentType, ValidationResult
)
from ..config import ProcessingConfig, SystemConfig
from ..processors import DocumentRouter, ProcessingError
from .progress_tracker import ProgressTracker, ProcessingStatus
from .resource_manager import ResourceManager, ResourceConstraints


logger = logging.getLogger(__name__)


class BatchProcessingError(Exception):
    """Exception raised during batch processing operations."""
    
    def __init__(self, message: str, failed_files: List[str] = None, cause: Exception = None):
        super().__init__(message)
        self.failed_files = failed_files or []
        self.cause = cause


@dataclass
class BatchResult:
    """Result of batch processing operation."""
    success: bool
    total_files: int
    processed_files: int
    failed_files: int
    processing_time: float
    successful_documents: List[DocumentContent] = field(default_factory=list)
    failed_documents: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class FileProcessingTask:
    """Individual file processing task."""
    file_path: str
    priority: int = 0
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


class BatchProcessor:
    """
    Coordinates batch processing of multiple files with progress tracking,
    error handling, and resource management.
    """
    
    def __init__(
        self,
        config: SystemConfig,
        document_router: Optional[DocumentRouter] = None,
        resource_manager: Optional[ResourceManager] = None,
        progress_callback: Optional[Callable[[BatchProcessingStatus], None]] = None
    ):
        """
        Initialize batch processor.
        
        Args:
            config: System configuration
            document_router: Document router for file processing
            resource_manager: Resource manager for optimization
            progress_callback: Optional callback for progress updates
        """
        self.config = config
        self.processing_config = config.processing
        self.document_router = document_router or DocumentRouter(self.processing_config)
        self.resource_manager = resource_manager or ResourceManager(config)
        self.progress_callback = progress_callback
        
        # Initialize progress tracker
        self.progress_tracker = ProgressTracker()
        
        # Processing state
        self._processing_lock = threading.Lock()
        self._is_processing = False
        self._should_stop = False
        
        # Statistics
        self._stats = {
            'total_batches': 0,
            'total_files_processed': 0,
            'total_processing_time': 0.0,
            'average_file_time': 0.0,
            'error_rate': 0.0
        }
        
        logger.info(f"BatchProcessor initialized with max_concurrent_files={self.processing_config.max_concurrent_files}")
    
    def process_files(
        self,
        file_paths: List[str],
        priority_order: bool = False,
        fail_fast: bool = False
    ) -> BatchResult:
        """
        Process multiple files in batch.
        
        Args:
            file_paths: List of file paths to process
            priority_order: Whether to process files in priority order
            fail_fast: Whether to stop on first error
            
        Returns:
            BatchResult with processing results and statistics
            
        Raises:
            BatchProcessingError: If batch processing fails critically
        """
        if not file_paths:
            return BatchResult(
                success=True,
                total_files=0,
                processed_files=0,
                failed_files=0,
                processing_time=0.0
            )
        
        with self._processing_lock:
            if self._is_processing:
                raise BatchProcessingError("Batch processing already in progress")
            self._is_processing = True
            self._should_stop = False
        
        try:
            return self._execute_batch_processing(file_paths, priority_order, fail_fast)
        finally:
            with self._processing_lock:
                self._is_processing = False
    
    def process_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        file_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> BatchResult:
        """
        Process all supported files in a directory.
        
        Args:
            directory_path: Path to directory to process
            recursive: Whether to process subdirectories
            file_patterns: Optional list of file patterns to include
            exclude_patterns: Optional list of file patterns to exclude
            
        Returns:
            BatchResult with processing results
        """
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            raise BatchProcessingError(f"Directory does not exist: {directory_path}")
        
        # Discover files
        file_paths = self._discover_files(
            directory, recursive, file_patterns, exclude_patterns
        )
        
        logger.info(f"Discovered {len(file_paths)} files in {directory_path}")
        
        return self.process_files(file_paths)
    
    def stop_processing(self) -> None:
        """Stop current batch processing operation."""
        with self._processing_lock:
            self._should_stop = True
        logger.info("Batch processing stop requested")
    
    def is_processing(self) -> bool:
        """Check if batch processing is currently active."""
        with self._processing_lock:
            return self._is_processing
    
    def get_progress(self) -> BatchProcessingStatus:
        """Get current processing progress."""
        return self.progress_tracker.get_status()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        return self._stats.copy()
    
    def _execute_batch_processing(
        self,
        file_paths: List[str],
        priority_order: bool,
        fail_fast: bool
    ) -> BatchResult:
        """Execute the actual batch processing."""
        start_time = time.time()
        
        # Create processing tasks
        tasks = self._create_processing_tasks(file_paths, priority_order)
        
        # Initialize progress tracking
        self.progress_tracker.start_batch(len(tasks))
        self._notify_progress()
        
        # Process files
        successful_documents = []
        failed_documents = []
        errors = []
        warnings = []
        
        try:
            # Determine optimal concurrency
            max_workers = self.resource_manager.get_optimal_concurrency(
                len(tasks), 'file_processing'
            )
            
            logger.info(f"Processing {len(tasks)} files with {max_workers} workers")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(self._process_single_file, task): task
                    for task in tasks
                }
                
                # Process completed tasks
                for future in as_completed(future_to_task):
                    if self._should_stop:
                        logger.info("Stopping batch processing due to stop request")
                        break
                    
                    task = future_to_task[future]
                    
                    try:
                        result = future.result()
                        
                        if result.success:
                            successful_documents.append(result.document_content)
                            self.progress_tracker.mark_file_completed(task.file_path, True)
                        else:
                            failed_documents.append({
                                'file_path': task.file_path,
                                'error': result.error_message,
                                'retry_count': task.retry_count
                            })
                            errors.append(f"{task.file_path}: {result.error_message}")
                            self.progress_tracker.mark_file_completed(task.file_path, False)
                            
                            if fail_fast:
                                logger.error(f"Failing fast due to error in {task.file_path}")
                                break
                    
                    except Exception as e:
                        error_msg = f"Unexpected error processing {task.file_path}: {str(e)}"
                        errors.append(error_msg)
                        failed_documents.append({
                            'file_path': task.file_path,
                            'error': error_msg,
                            'retry_count': task.retry_count
                        })
                        self.progress_tracker.mark_file_completed(task.file_path, False)
                        logger.exception(f"Unexpected error processing {task.file_path}")
                        
                        if fail_fast:
                            break
                    
                    self._notify_progress()
        
        except Exception as e:
            logger.exception("Critical error during batch processing")
            raise BatchProcessingError(f"Batch processing failed: {str(e)}", cause=e)
        
        finally:
            self.progress_tracker.complete_batch()
        
        # Calculate results
        processing_time = time.time() - start_time
        total_files = len(tasks)
        processed_files = len(successful_documents)
        failed_files = len(failed_documents)
        
        # Update statistics
        self._update_statistics(total_files, processed_files, processing_time)
        
        # Create result
        result = BatchResult(
            success=failed_files == 0 or not fail_fast,
            total_files=total_files,
            processed_files=processed_files,
            failed_files=failed_files,
            processing_time=processing_time,
            successful_documents=successful_documents,
            failed_documents=failed_documents,
            errors=errors,
            warnings=warnings
        )
        
        logger.info(
            f"Batch processing completed: {processed_files}/{total_files} files processed "
            f"in {processing_time:.2f}s"
        )
        
        return result
    
    def _create_processing_tasks(
        self,
        file_paths: List[str],
        priority_order: bool
    ) -> List[FileProcessingTask]:
        """Create processing tasks from file paths."""
        tasks = []
        
        for i, file_path in enumerate(file_paths):
            # Assign priority based on file characteristics
            priority = self._calculate_file_priority(file_path) if priority_order else i
            
            task = FileProcessingTask(
                file_path=file_path,
                priority=priority,
                metadata={'original_index': i}
            )
            tasks.append(task)
        
        # Sort by priority if requested
        if priority_order:
            tasks.sort(key=lambda t: t.priority, reverse=True)
        
        return tasks
    
    def _calculate_file_priority(self, file_path: str) -> int:
        """Calculate processing priority for a file."""
        path = Path(file_path)
        priority = 0
        
        # Prioritize by file size (smaller files first for quick wins)
        try:
            file_size = path.stat().st_size
            if file_size < 1024 * 1024:  # < 1MB
                priority += 100
            elif file_size < 10 * 1024 * 1024:  # < 10MB
                priority += 50
        except:
            pass
        
        # Prioritize by file type (faster processing types first)
        extension = path.suffix.lower()
        if extension in ['.txt', '.md']:
            priority += 75
        elif extension in ['.pdf']:
            priority += 50
        elif extension in ['.docx']:
            priority += 40
        elif extension in ['.png', '.jpg', '.jpeg']:
            priority += 30
        elif extension in ['.mp3', '.wav', '.m4a']:
            priority += 10
        
        return priority
    
    def _process_single_file(self, task: FileProcessingTask) -> ProcessingResult:
        """Process a single file with retry logic."""
        file_path = task.file_path
        
        for attempt in range(task.max_retries + 1):
            try:
                self.progress_tracker.start_file_processing(file_path)
                
                # Validate file
                validation = self.document_router.validate_file(file_path)
                if not validation.is_valid:
                    return ProcessingResult(
                        success=False,
                        error_message=validation.error_message
                    )
                
                # Route to appropriate processor
                processor = self.document_router.route_document(file_path)
                
                # Process document
                start_time = time.time()
                document_content = processor.process_document(file_path)
                processing_time = time.time() - start_time
                
                logger.debug(f"Processed {file_path} in {processing_time:.2f}s")
                
                return ProcessingResult(
                    success=True,
                    document_content=document_content,
                    processing_time=processing_time,
                    chunks_created=len(document_content.chunks)
                )
            
            except ProcessingError as e:
                task.retry_count = attempt + 1
                
                if attempt < task.max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(
                        f"Processing failed for {file_path} (attempt {attempt + 1}), "
                        f"retrying in {wait_time}s: {str(e)}"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Processing failed for {file_path} after {task.max_retries + 1} attempts")
                    return ProcessingResult(
                        success=False,
                        error_message=f"Processing failed after {task.max_retries + 1} attempts: {str(e)}"
                    )
            
            except Exception as e:
                task.retry_count = attempt + 1
                logger.exception(f"Unexpected error processing {file_path}")
                
                if attempt < task.max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(f"Retrying {file_path} in {wait_time}s")
                    time.sleep(wait_time)
                    continue
                else:
                    return ProcessingResult(
                        success=False,
                        error_message=f"Unexpected error after {task.max_retries + 1} attempts: {str(e)}"
                    )
        
        # Should never reach here
        return ProcessingResult(
            success=False,
            error_message="Processing failed for unknown reason"
        )
    
    def _discover_files(
        self,
        directory: Path,
        recursive: bool,
        file_patterns: Optional[List[str]],
        exclude_patterns: Optional[List[str]]
    ) -> List[str]:
        """Discover files in directory matching criteria."""
        files = []
        supported_extensions = set(f".{fmt}" for fmt in self.processing_config.supported_formats)
        
        # Get all files
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        for file_path in directory.glob(pattern):
            if not file_path.is_file():
                continue
            
            # Check extension
            if file_path.suffix.lower() not in supported_extensions:
                continue
            
            # Check file patterns
            if file_patterns:
                if not any(file_path.match(pattern) for pattern in file_patterns):
                    continue
            
            # Check exclude patterns
            if exclude_patterns:
                if any(file_path.match(pattern) for pattern in exclude_patterns):
                    continue
            
            files.append(str(file_path))
        
        return files
    
    def _notify_progress(self) -> None:
        """Notify progress callback if available."""
        if self.progress_callback:
            try:
                status = self.progress_tracker.get_status()
                self.progress_callback(status)
            except Exception as e:
                logger.warning(f"Progress callback failed: {str(e)}")
    
    def _update_statistics(
        self,
        total_files: int,
        processed_files: int,
        processing_time: float
    ) -> None:
        """Update processing statistics."""
        self._stats['total_batches'] += 1
        self._stats['total_files_processed'] += processed_files
        self._stats['total_processing_time'] += processing_time
        
        if self._stats['total_files_processed'] > 0:
            self._stats['average_file_time'] = (
                self._stats['total_processing_time'] / self._stats['total_files_processed']
            )
        
        if total_files > 0:
            failed_files = total_files - processed_files
            self._stats['error_rate'] = failed_files / total_files