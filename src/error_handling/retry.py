"""Retry logic with exponential backoff for error recovery."""

import time
import random
import logging
from typing import Callable, Any, Optional, Type, Union, List
from functools import wraps
from dataclasses import dataclass

from .exceptions import BaseRAGError, ErrorSeverity


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_factor: float = 1.0


class RetryManager:
    """Manages retry logic with exponential backoff."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.logger = logging.getLogger(__name__)
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = self.config.base_delay * (
            self.config.exponential_base ** (attempt - 1)
        ) * self.config.backoff_factor
        
        # Cap at max delay
        delay = min(delay, self.config.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            jitter_range = delay * 0.1
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)
    
    def should_retry(
        self,
        error: Exception,
        attempt: int,
        retryable_exceptions: Optional[List[Type[Exception]]] = None
    ) -> bool:
        """Determine if an error should trigger a retry."""
        if attempt >= self.config.max_attempts:
            return False
        
        # Check if error is retryable
        if retryable_exceptions:
            if not any(isinstance(error, exc_type) for exc_type in retryable_exceptions):
                return False
        
        # For RAG errors, check if they're marked as recoverable
        if isinstance(error, BaseRAGError):
            if not error.recoverable:
                return False
            # Don't retry critical errors
            if error.severity == ErrorSeverity.CRITICAL:
                return False
        
        return True
    
    def retry_with_backoff(
        self,
        func: Callable,
        *args,
        retryable_exceptions: Optional[List[Type[Exception]]] = None,
        **kwargs
    ) -> Any:
        """Execute function with retry logic."""
        last_error = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                
                if not self.should_retry(e, attempt, retryable_exceptions):
                    self.logger.error(
                        f"Function {func.__name__} failed after {attempt} attempts: {e}"
                    )
                    raise e
                
                if attempt < self.config.max_attempts:
                    delay = self.calculate_delay(attempt)
                    self.logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    time.sleep(delay)
        
        # If we get here, all attempts failed
        self.logger.error(
            f"Function {func.__name__} failed after {self.config.max_attempts} attempts"
        )
        raise last_error


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[List[Type[Exception]]] = None
):
    """Decorator for adding retry logic to functions."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            config = RetryConfig(
                max_attempts=max_attempts,
                base_delay=base_delay,
                max_delay=max_delay,
                exponential_base=exponential_base,
                jitter=jitter
            )
            
            retry_manager = RetryManager(config)
            return retry_manager.retry_with_backoff(
                func, *args, retryable_exceptions=retryable_exceptions, **kwargs
            )
        
        return wrapper
    return decorator


def async_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[List[Type[Exception]]] = None
):
    """Decorator for adding retry logic to async functions."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            import asyncio
            
            config = RetryConfig(
                max_attempts=max_attempts,
                base_delay=base_delay,
                max_delay=max_delay,
                exponential_base=exponential_base,
                jitter=jitter
            )
            
            retry_manager = RetryManager(config)
            last_error = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    
                    if not retry_manager.should_retry(e, attempt, retryable_exceptions):
                        raise e
                    
                    if attempt < max_attempts:
                        delay = retry_manager.calculate_delay(attempt)
                        await asyncio.sleep(delay)
            
            raise last_error
        
        return wrapper
    return decorator