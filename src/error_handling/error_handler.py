"""Main error handler class with categorized error handling strategies."""

import logging
import traceback
from typing import Any, Dict, Optional, Callable, List, Union
from contextlib import contextmanager

from .exceptions import (
    BaseRAGError, ProcessingError, ModelError, StorageError,
    RetrievalError, ValidationError, ConfigurationError, ResourceError,
    ErrorCategory, ErrorSeverity
)
from .recovery import RecoveryManager, GracefulDegradation
from .retry import RetryManager, RetryConfig


class ErrorHandler:
    """Main error handler with categorized error handling strategies."""
    
    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        enable_graceful_degradation: bool = True
    ):
        self.logger = logging.getLogger(__name__)
        self.recovery_manager = RecoveryManager()
        self.retry_manager = RetryManager(retry_config)
        self.degradation = GracefulDegradation() if enable_graceful_degradation else None
        
        # Error statistics
        self._error_counts: Dict[str, int] = {}
        self._error_history: List[Dict[str, Any]] = []
        
        # Error handlers for different categories
        self._category_handlers: Dict[ErrorCategory, Callable] = {
            ErrorCategory.PROCESSING: self._handle_processing_error,
            ErrorCategory.MODEL: self._handle_model_error,
            ErrorCategory.STORAGE: self._handle_storage_error,
            ErrorCategory.RETRIEVAL: self._handle_retrieval_error,
            ErrorCategory.VALIDATION: self._handle_validation_error,
            ErrorCategory.CONFIGURATION: self._handle_configuration_error,
            ErrorCategory.RESOURCE: self._handle_resource_error,
        }
    
    def handle_error(
        self,
        error: Union[Exception, BaseRAGError],
        context: Optional[Dict[str, Any]] = None,
        operation: Optional[str] = None
    ) -> Any:
        """Main error handling entry point."""
        
        # Convert regular exceptions to RAG errors if needed
        if not isinstance(error, BaseRAGError):
            error = self._convert_to_rag_error(error, context)
        
        # Log the error with context
        self._log_error_with_context(error, context, operation)
        
        # Update error statistics
        self._update_error_statistics(error)
        
        # Get appropriate handler for error category
        handler = self._category_handlers.get(error.category, self._handle_unknown_error)
        
        try:
            return handler(error, context)
        except Exception as handler_error:
            self.logger.error(f"Error handler failed: {handler_error}")
            # Fallback to basic recovery
            return self.recovery_manager.execute_recovery(error, context)
    
    def _convert_to_rag_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> BaseRAGError:
        """Convert regular exceptions to RAG errors."""
        
        # Map common exception types to RAG error categories
        if isinstance(error, (FileNotFoundError, PermissionError, OSError)):
            return ProcessingError(
                message=str(error),
                context=context,
                original_error=error
            )
        elif isinstance(error, (ImportError, ModuleNotFoundError)):
            return ModelError(
                message=f"Model dependency error: {error}",
                context=context,
                original_error=error
            )
        elif isinstance(error, (ValueError, TypeError)):
            return ValidationError(
                message=str(error),
                context=context,
                original_error=error
            )
        elif isinstance(error, MemoryError):
            return ResourceError(
                message="Insufficient memory",
                resource_type="memory",
                context=context,
                original_error=error
            )
        else:
            # Generic processing error for unknown exceptions
            return ProcessingError(
                message=f"Unexpected error: {error}",
                severity=ErrorSeverity.HIGH,
                context=context,
                original_error=error
            )
    
    def _log_error_with_context(
        self,
        error: BaseRAGError,
        context: Optional[Dict[str, Any]] = None,
        operation: Optional[str] = None
    ):
        """Log error with full context information."""
        
        log_data = {
            'error_type': type(error).__name__,
            'category': error.category.value,
            'severity': error.severity.value,
            'message': error.message,
            'recoverable': error.recoverable,
            'operation': operation,
            'context': error.context,
            'additional_context': context
        }
        
        if error.original_error:
            log_data['original_error'] = str(error.original_error)
            log_data['traceback'] = traceback.format_exception(
                type(error.original_error),
                error.original_error,
                error.original_error.__traceback__
            )
        
        # Choose log level based on severity
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical("Critical error occurred", extra=log_data)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error("High severity error occurred", extra=log_data)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning("Medium severity error occurred", extra=log_data)
        else:
            self.logger.info("Low severity error occurred", extra=log_data)
    
    def _update_error_statistics(self, error: BaseRAGError):
        """Update error statistics for monitoring."""
        error_key = f"{error.category.value}_{error.severity.value}"
        self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1
        
        # Keep error history (limited to last 1000 errors)
        self._error_history.append({
            'timestamp': self._get_timestamp(),
            'category': error.category.value,
            'severity': error.severity.value,
            'message': error.message,
            'recoverable': error.recoverable
        })
        
        if len(self._error_history) > 1000:
            self._error_history = self._error_history[-1000:]
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for error logging."""
        import datetime
        return datetime.datetime.now().isoformat()
    
    def _handle_processing_error(
        self,
        error: ProcessingError,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Handle document processing errors."""
        
        # For processing errors, we often want to skip the problematic file
        # and continue with others
        if error.severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]:
            self.logger.warning(f"Skipping file due to processing error: {error.message}")
            return None
        
        # For high severity errors, try recovery
        return self.recovery_manager.execute_recovery(error, context)
    
    def _handle_model_error(
        self,
        error: ModelError,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Handle model loading and inference errors."""
        
        # Enable degradation mode for the affected model
        if self.degradation and 'model_name' in error.context:
            model_name = error.context['model_name']
            self.degradation.enable_degradation_mode(
                f"model_{model_name}",
                f"Model error: {error.message}"
            )
        
        # Try to recover with fallback model or method
        return self.recovery_manager.execute_recovery(error, context)
    
    def _handle_storage_error(
        self,
        error: StorageError,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Handle storage operation errors."""
        
        # Storage errors are critical for data integrity
        if error.severity >= ErrorSeverity.HIGH:
            self.logger.error("Critical storage error - may affect data integrity")
        
        return self.recovery_manager.execute_recovery(error, context)
    
    def _handle_retrieval_error(
        self,
        error: RetrievalError,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Handle retrieval operation errors."""
        
        # For retrieval errors, we can often return empty results
        # and let the system continue
        if error.severity <= ErrorSeverity.MEDIUM:
            self.logger.warning("Retrieval failed - returning empty results")
            return []
        
        return self.recovery_manager.execute_recovery(error, context)
    
    def _handle_validation_error(
        self,
        error: ValidationError,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Handle data validation errors."""
        
        # Validation errors are usually not recoverable
        # Log the error and skip the invalid data
        self.logger.error(f"Data validation failed: {error.message}")
        return None
    
    def _handle_configuration_error(
        self,
        error: ConfigurationError,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Handle configuration errors."""
        
        # Configuration errors often require manual intervention
        # Try to use default values if possible
        if error.severity <= ErrorSeverity.MEDIUM:
            self.logger.warning("Using default configuration due to error")
            # Return signal to use defaults
            return {"use_defaults": True, "reason": error.message}
        
        return self.recovery_manager.execute_recovery(error, context)
    
    def _handle_resource_error(
        self,
        error: ResourceError,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Handle resource-related errors."""
        
        # Enable resource-constrained mode
        if self.degradation:
            self.degradation.enable_degradation_mode(
                "resource_usage",
                f"Resource constraint: {error.message}"
            )
        
        return self.recovery_manager.execute_recovery(error, context)
    
    def _handle_unknown_error(
        self,
        error: BaseRAGError,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Handle unknown error categories."""
        
        self.logger.error(f"Unknown error category: {error.category}")
        return self.recovery_manager.execute_recovery(error, context)
    
    @contextmanager
    def error_context(self, operation: str, **context_data):
        """Context manager for handling errors in operations."""
        try:
            yield
        except Exception as e:
            self.handle_error(e, context_data, operation)
            raise
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        return {
            'error_counts': self._error_counts.copy(),
            'total_errors': sum(self._error_counts.values()),
            'recent_errors': self._error_history[-10:] if self._error_history else [],
            'degradation_status': self.degradation.get_degradation_status() if self.degradation else {}
        }
    
    def reset_error_statistics(self):
        """Reset error statistics."""
        self._error_counts.clear()
        self._error_history.clear()
        self.logger.info("Error statistics reset")
    
    def register_custom_handler(self, category: ErrorCategory, handler: Callable):
        """Register a custom error handler for a category."""
        self._category_handlers[category] = handler
        self.logger.info(f"Registered custom handler for {category.value}")
    
    def with_retry(
        self,
        func: Callable,
        *args,
        retryable_exceptions: Optional[List[type]] = None,
        **kwargs
    ) -> Any:
        """Execute function with retry logic."""
        return self.retry_manager.retry_with_backoff(
            func, *args, retryable_exceptions=retryable_exceptions, **kwargs
        )