"""Custom exception classes for the multimodal RAG system."""

from typing import Optional, Dict, Any
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    PROCESSING = "processing"
    MODEL = "model"
    STORAGE = "storage"
    RETRIEVAL = "retrieval"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"
    NETWORK = "network"


class BaseRAGError(Exception):
    """Base exception class for RAG system errors."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.recoverable = recoverable
        self.original_error = original_error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        return {
            'message': self.message,
            'category': self.category.value,
            'severity': self.severity.value,
            'context': self.context,
            'recoverable': self.recoverable,
            'original_error': str(self.original_error) if self.original_error else None
        }


class ProcessingError(BaseRAGError):
    """Error during document processing."""
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        processor_type: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        context = context or {}
        if file_path:
            context['file_path'] = file_path
        if processor_type:
            context['processor_type'] = processor_type
            
        super().__init__(
            message=message,
            category=ErrorCategory.PROCESSING,
            severity=severity,
            context=context,
            recoverable=True,
            original_error=original_error
        )


class ModelError(BaseRAGError):
    """Error during model loading or inference."""
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        model_type: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        context = context or {}
        if model_name:
            context['model_name'] = model_name
        if model_type:
            context['model_type'] = model_type
            
        super().__init__(
            message=message,
            category=ErrorCategory.MODEL,
            severity=severity,
            context=context,
            recoverable=True,
            original_error=original_error
        )


class StorageError(BaseRAGError):
    """Error during storage operations."""
    
    def __init__(
        self,
        message: str,
        storage_type: Optional[str] = None,
        operation: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        context = context or {}
        if storage_type:
            context['storage_type'] = storage_type
        if operation:
            context['operation'] = operation
            
        super().__init__(
            message=message,
            category=ErrorCategory.STORAGE,
            severity=severity,
            context=context,
            recoverable=True,
            original_error=original_error
        )


class RetrievalError(BaseRAGError):
    """Error during retrieval operations."""
    
    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        retrieval_type: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        context = context or {}
        if query:
            context['query'] = query
        if retrieval_type:
            context['retrieval_type'] = retrieval_type
            
        super().__init__(
            message=message,
            category=ErrorCategory.RETRIEVAL,
            severity=severity,
            context=context,
            recoverable=True,
            original_error=original_error
        )


class ValidationError(BaseRAGError):
    """Error during data validation."""
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        expected_type: Optional[str] = None,
        actual_value: Optional[Any] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        context = context or {}
        if field_name:
            context['field_name'] = field_name
        if expected_type:
            context['expected_type'] = expected_type
        if actual_value is not None:
            context['actual_value'] = str(actual_value)
            
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=severity,
            context=context,
            recoverable=False,
            original_error=original_error
        )


class ConfigurationError(BaseRAGError):
    """Error in system configuration."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_file: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        context = context or {}
        if config_key:
            context['config_key'] = config_key
        if config_file:
            context['config_file'] = config_file
            
        super().__init__(
            message=message,
            category=ErrorCategory.CONFIGURATION,
            severity=severity,
            context=context,
            recoverable=False,
            original_error=original_error
        )


class ResourceError(BaseRAGError):
    """Error related to system resources."""
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        available: Optional[str] = None,
        required: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        context = context or {}
        if resource_type:
            context['resource_type'] = resource_type
        if available:
            context['available'] = available
        if required:
            context['required'] = required
            
        super().__init__(
            message=message,
            category=ErrorCategory.RESOURCE,
            severity=severity,
            context=context,
            recoverable=True,
            original_error=original_error
        )