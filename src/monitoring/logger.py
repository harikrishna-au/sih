"""Structured logging implementation with configurable levels."""

import logging
import logging.config
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum


class LogLevel(Enum):
    """Log levels for the system."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogConfig:
    """Configuration for structured logging."""
    level: LogLevel = LogLevel.INFO
    log_file: Optional[str] = None
    max_file_size_mb: int = 100
    backup_count: int = 5
    format_type: str = "json"  # json or text
    include_timestamp: bool = True
    include_process_info: bool = True
    include_thread_info: bool = False
    console_output: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def __init__(self, include_timestamp: bool = True, include_process_info: bool = True):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_process_info = include_process_info
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        
        # Base log entry
        log_entry = {
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        # Add timestamp if requested
        if self.include_timestamp:
            log_entry['timestamp'] = datetime.fromtimestamp(record.created).isoformat()
        
        # Add process info if requested
        if self.include_process_info:
            log_entry['process'] = record.process
            log_entry['process_name'] = record.processName
        
        # Add thread info if available
        if hasattr(record, 'thread') and record.thread:
            log_entry['thread'] = record.thread
            log_entry['thread_name'] = record.threadName
        
        # Add module and function info
        log_entry['module'] = record.module
        log_entry['function'] = record.funcName
        log_entry['line'] = record.lineno
        
        # Add extra fields from the record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                extra_fields[key] = value
        
        if extra_fields:
            log_entry['extra'] = extra_fields
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)


class StructuredLogger:
    """Structured logger with configurable levels and output formats."""
    
    def __init__(self, name: str, config: Optional[LogConfig] = None):
        self.name = name
        self.config = config or LogConfig()
        self.logger = logging.getLogger(name)
        self._setup_logger()
        
        # Performance tracking
        self._operation_times: Dict[str, float] = {}
        self._operation_counts: Dict[str, int] = {}
    
    def _setup_logger(self):
        """Setup logger with handlers and formatters."""
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set log level
        self.logger.setLevel(getattr(logging, self.config.level.value))
        
        # Create formatters
        if self.config.format_type == "json":
            formatter = StructuredFormatter(
                include_timestamp=self.config.include_timestamp,
                include_process_info=self.config.include_process_info
            )
        else:
            # Text formatter
            format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            if self.config.include_process_info:
                format_str = '%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s'
            formatter = logging.Formatter(format_str)
        
        # Console handler
        if self.config.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.log_file:
            from logging.handlers import RotatingFileHandler
            
            # Ensure log directory exists
            log_path = Path(self.config.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                self.config.log_file,
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with structured data."""
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data."""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data."""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with structured data."""
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with structured data."""
        self.logger.critical(message, extra=kwargs)
    
    def log_operation_start(self, operation: str, **context):
        """Log the start of an operation."""
        self.info(f"Starting operation: {operation}", 
                 operation=operation, 
                 operation_status="started",
                 **context)
        
        # Track operation start time
        import time
        self._operation_times[operation] = time.time()
    
    def log_operation_end(self, operation: str, success: bool = True, **context):
        """Log the end of an operation with timing."""
        import time
        
        # Calculate duration
        duration = None
        if operation in self._operation_times:
            duration = time.time() - self._operation_times[operation]
            del self._operation_times[operation]
        
        # Update operation counts
        self._operation_counts[operation] = self._operation_counts.get(operation, 0) + 1
        
        status = "completed" if success else "failed"
        log_data = {
            'operation': operation,
            'operation_status': status,
            'duration_seconds': duration,
            'operation_count': self._operation_counts[operation],
            **context
        }
        
        if success:
            self.info(f"Operation completed: {operation}", **log_data)
        else:
            self.error(f"Operation failed: {operation}", **log_data)
    
    def log_performance_metric(self, metric_name: str, value: Union[int, float], unit: str = "", **context):
        """Log a performance metric."""
        self.info(f"Performance metric: {metric_name}",
                 metric_name=metric_name,
                 metric_value=value,
                 metric_unit=unit,
                 metric_type="performance",
                 **context)
    
    def log_error_with_context(self, error: Exception, operation: Optional[str] = None, **context):
        """Log an error with full context and stack trace."""
        import traceback
        
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'operation': operation,
            'traceback': traceback.format_exc(),
            **context
        }
        
        self.error(f"Error occurred: {error}", **error_data)
    
    def log_system_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log a system event with structured data."""
        self.info(f"System event: {event_type}",
                 event_type=event_type,
                 event_data=event_data)
    
    def get_operation_statistics(self) -> Dict[str, Any]:
        """Get operation statistics for monitoring."""
        return {
            'operation_counts': self._operation_counts.copy(),
            'active_operations': list(self._operation_times.keys()),
            'total_operations': sum(self._operation_counts.values())
        }
    
    def reset_statistics(self):
        """Reset operation statistics."""
        self._operation_counts.clear()
        self._operation_times.clear()
        self.info("Logger statistics reset")
    
    def update_config(self, config: LogConfig):
        """Update logger configuration."""
        self.config = config
        self._setup_logger()
        self.info("Logger configuration updated", config=config.to_dict())
    
    def set_level(self, level: LogLevel):
        """Set logging level."""
        self.config.level = level
        self.logger.setLevel(getattr(logging, level.value))
        self.info(f"Log level changed to {level.value}")


class LoggerManager:
    """Manager for multiple structured loggers."""
    
    def __init__(self, default_config: Optional[LogConfig] = None):
        self.default_config = default_config or LogConfig()
        self._loggers: Dict[str, StructuredLogger] = {}
    
    def get_logger(self, name: str, config: Optional[LogConfig] = None) -> StructuredLogger:
        """Get or create a structured logger."""
        if name not in self._loggers:
            logger_config = config or self.default_config
            self._loggers[name] = StructuredLogger(name, logger_config)
        return self._loggers[name]
    
    def configure_all_loggers(self, config: LogConfig):
        """Configure all existing loggers with new config."""
        self.default_config = config
        for logger in self._loggers.values():
            logger.update_config(config)
    
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics from all loggers."""
        return {
            name: logger.get_operation_statistics()
            for name, logger in self._loggers.items()
        }
    
    def reset_all_statistics(self):
        """Reset statistics for all loggers."""
        for logger in self._loggers.values():
            logger.reset_statistics()


# Global logger manager instance
_logger_manager = LoggerManager()


def get_logger(name: str, config: Optional[LogConfig] = None) -> StructuredLogger:
    """Get a structured logger instance."""
    return _logger_manager.get_logger(name, config)


def configure_logging(config: LogConfig):
    """Configure global logging settings."""
    _logger_manager.configure_all_loggers(config)


def get_logging_statistics() -> Dict[str, Dict[str, Any]]:
    """Get statistics from all loggers."""
    return _logger_manager.get_all_statistics()