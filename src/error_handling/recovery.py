"""Recovery mechanisms for graceful degradation."""

import logging
from typing import Any, Dict, Optional, Callable, List
from enum import Enum
from dataclasses import dataclass

from .exceptions import BaseRAGError, ErrorCategory, ErrorSeverity


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    FALLBACK = "fallback"
    SKIP = "skip"
    RETRY = "retry"
    DEGRADE = "degrade"
    ABORT = "abort"


@dataclass
class RecoveryAction:
    """Represents a recovery action to take."""
    strategy: RecoveryStrategy
    fallback_value: Optional[Any] = None
    fallback_function: Optional[Callable] = None
    context: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class RecoveryManager:
    """Manages recovery strategies for different error scenarios."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._recovery_strategies: Dict[ErrorCategory, Dict[ErrorSeverity, RecoveryAction]] = {}
        self._setup_default_strategies()
    
    def _setup_default_strategies(self):
        """Set up default recovery strategies for different error types."""
        
        # Processing errors - usually recoverable
        self._recovery_strategies[ErrorCategory.PROCESSING] = {
            ErrorSeverity.LOW: RecoveryAction(
                strategy=RecoveryStrategy.SKIP,
                message="Skipping file due to minor processing error"
            ),
            ErrorSeverity.MEDIUM: RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                message="Retrying processing with fallback method"
            ),
            ErrorSeverity.HIGH: RecoveryAction(
                strategy=RecoveryStrategy.SKIP,
                message="Skipping file due to severe processing error"
            ),
            ErrorSeverity.CRITICAL: RecoveryAction(
                strategy=RecoveryStrategy.ABORT,
                message="Critical processing error - aborting operation"
            )
        }
        
        # Model errors - often require fallback
        self._recovery_strategies[ErrorCategory.MODEL] = {
            ErrorSeverity.LOW: RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                message="Retrying model operation"
            ),
            ErrorSeverity.MEDIUM: RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                message="Using fallback model or method"
            ),
            ErrorSeverity.HIGH: RecoveryAction(
                strategy=RecoveryStrategy.DEGRADE,
                message="Degrading to simpler processing method"
            ),
            ErrorSeverity.CRITICAL: RecoveryAction(
                strategy=RecoveryStrategy.ABORT,
                message="Critical model error - cannot continue"
            )
        }
        
        # Storage errors - critical for data integrity
        self._recovery_strategies[ErrorCategory.STORAGE] = {
            ErrorSeverity.LOW: RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                message="Retrying storage operation"
            ),
            ErrorSeverity.MEDIUM: RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                message="Retrying with backup storage method"
            ),
            ErrorSeverity.HIGH: RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                message="Using alternative storage backend"
            ),
            ErrorSeverity.CRITICAL: RecoveryAction(
                strategy=RecoveryStrategy.ABORT,
                message="Critical storage error - data integrity at risk"
            )
        }
        
        # Retrieval errors - can often continue with degraded results
        self._recovery_strategies[ErrorCategory.RETRIEVAL] = {
            ErrorSeverity.LOW: RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                fallback_value=[],
                message="Returning empty results due to minor retrieval error"
            ),
            ErrorSeverity.MEDIUM: RecoveryAction(
                strategy=RecoveryStrategy.DEGRADE,
                message="Using simpler retrieval method"
            ),
            ErrorSeverity.HIGH: RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                fallback_value=[],
                message="Retrieval failed - returning empty results"
            ),
            ErrorSeverity.CRITICAL: RecoveryAction(
                strategy=RecoveryStrategy.ABORT,
                message="Critical retrieval error"
            )
        }
        
        # Validation errors - usually not recoverable
        self._recovery_strategies[ErrorCategory.VALIDATION] = {
            ErrorSeverity.LOW: RecoveryAction(
                strategy=RecoveryStrategy.SKIP,
                message="Skipping invalid data"
            ),
            ErrorSeverity.MEDIUM: RecoveryAction(
                strategy=RecoveryStrategy.SKIP,
                message="Validation failed - skipping item"
            ),
            ErrorSeverity.HIGH: RecoveryAction(
                strategy=RecoveryStrategy.ABORT,
                message="Validation error - cannot proceed safely"
            ),
            ErrorSeverity.CRITICAL: RecoveryAction(
                strategy=RecoveryStrategy.ABORT,
                message="Critical validation error"
            )
        }
        
        # Configuration errors - usually require manual intervention
        self._recovery_strategies[ErrorCategory.CONFIGURATION] = {
            ErrorSeverity.LOW: RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK,
                message="Using default configuration value"
            ),
            ErrorSeverity.MEDIUM: RecoveryAction(
                strategy=RecoveryStrategy.DEGRADE,
                message="Using minimal configuration"
            ),
            ErrorSeverity.HIGH: RecoveryAction(
                strategy=RecoveryStrategy.ABORT,
                message="Configuration error - manual intervention required"
            ),
            ErrorSeverity.CRITICAL: RecoveryAction(
                strategy=RecoveryStrategy.ABORT,
                message="Critical configuration error"
            )
        }
        
        # Resource errors - often recoverable with degradation
        self._recovery_strategies[ErrorCategory.RESOURCE] = {
            ErrorSeverity.LOW: RecoveryAction(
                strategy=RecoveryStrategy.RETRY,
                message="Retrying with current resources"
            ),
            ErrorSeverity.MEDIUM: RecoveryAction(
                strategy=RecoveryStrategy.DEGRADE,
                message="Reducing resource usage"
            ),
            ErrorSeverity.HIGH: RecoveryAction(
                strategy=RecoveryStrategy.DEGRADE,
                message="Switching to low-resource mode"
            ),
            ErrorSeverity.CRITICAL: RecoveryAction(
                strategy=RecoveryStrategy.ABORT,
                message="Insufficient resources - cannot continue"
            )
        }
    
    def get_recovery_action(self, error: BaseRAGError) -> RecoveryAction:
        """Get the appropriate recovery action for an error."""
        category_strategies = self._recovery_strategies.get(error.category)
        if not category_strategies:
            # Default fallback strategy
            return RecoveryAction(
                strategy=RecoveryStrategy.SKIP,
                message=f"Unknown error category {error.category} - skipping"
            )
        
        action = category_strategies.get(error.severity)
        if not action:
            # Default to medium severity strategy
            action = category_strategies.get(ErrorSeverity.MEDIUM)
            if not action:
                return RecoveryAction(
                    strategy=RecoveryStrategy.SKIP,
                    message="No recovery strategy available - skipping"
                )
        
        return action
    
    def register_recovery_strategy(
        self,
        category: ErrorCategory,
        severity: ErrorSeverity,
        action: RecoveryAction
    ):
        """Register a custom recovery strategy."""
        if category not in self._recovery_strategies:
            self._recovery_strategies[category] = {}
        
        self._recovery_strategies[category][severity] = action
        self.logger.info(
            f"Registered recovery strategy for {category.value}/{severity.value}: {action.strategy.value}"
        )
    
    def execute_recovery(self, error: BaseRAGError, context: Optional[Dict[str, Any]] = None) -> Any:
        """Execute recovery action for an error."""
        action = self.get_recovery_action(error)
        
        self.logger.warning(
            f"Executing recovery strategy {action.strategy.value} for error: {error.message}"
        )
        
        if action.message:
            self.logger.info(action.message)
        
        # Execute the recovery strategy
        if action.strategy == RecoveryStrategy.FALLBACK:
            if action.fallback_function:
                try:
                    return action.fallback_function(error, context)
                except Exception as e:
                    self.logger.error(f"Fallback function failed: {e}")
                    return action.fallback_value
            return action.fallback_value
        
        elif action.strategy == RecoveryStrategy.SKIP:
            self.logger.info(f"Skipping operation due to error: {error.message}")
            return None
        
        elif action.strategy == RecoveryStrategy.RETRY:
            # This should be handled by the retry manager
            raise error
        
        elif action.strategy == RecoveryStrategy.DEGRADE:
            self.logger.warning("Degrading system performance due to error")
            # Return a signal that degraded mode should be used
            return {"degraded": True, "reason": error.message}
        
        elif action.strategy == RecoveryStrategy.ABORT:
            self.logger.error(f"Aborting operation due to critical error: {error.message}")
            raise error
        
        else:
            self.logger.error(f"Unknown recovery strategy: {action.strategy}")
            raise error


class GracefulDegradation:
    """Manages graceful degradation of system capabilities."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._degradation_modes: Dict[str, bool] = {}
        self._fallback_implementations: Dict[str, Callable] = {}
    
    def enable_degradation_mode(self, component: str, reason: str = ""):
        """Enable degradation mode for a component."""
        self._degradation_modes[component] = True
        self.logger.warning(
            f"Enabled degradation mode for {component}" + 
            (f": {reason}" if reason else "")
        )
    
    def disable_degradation_mode(self, component: str):
        """Disable degradation mode for a component."""
        if component in self._degradation_modes:
            del self._degradation_modes[component]
            self.logger.info(f"Disabled degradation mode for {component}")
    
    def is_degraded(self, component: str) -> bool:
        """Check if a component is in degradation mode."""
        return self._degradation_modes.get(component, False)
    
    def register_fallback(self, component: str, fallback_func: Callable):
        """Register a fallback implementation for a component."""
        self._fallback_implementations[component] = fallback_func
        self.logger.info(f"Registered fallback implementation for {component}")
    
    def get_fallback(self, component: str) -> Optional[Callable]:
        """Get fallback implementation for a component."""
        return self._fallback_implementations.get(component)
    
    def get_degradation_status(self) -> Dict[str, bool]:
        """Get current degradation status for all components."""
        return self._degradation_modes.copy()