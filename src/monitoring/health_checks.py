"""Health checks for all system components."""

import time
import threading
import traceback
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

from .logger import get_logger
from .metrics import MetricsCollector


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a system component."""
    component_name: str
    status: HealthStatus
    message: str
    last_check: datetime
    check_duration: float
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class SystemHealth:
    """Overall system health status."""
    overall_status: HealthStatus
    components: Dict[str, ComponentHealth]
    check_timestamp: datetime
    total_components: int
    healthy_components: int
    warning_components: int
    critical_components: int
    unknown_components: int


class HealthChecker:
    """Health checker for system components."""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self.logger = get_logger(__name__)
        self.metrics = metrics_collector or MetricsCollector()
        self._lock = threading.RLock()
        
        # Health check registry
        self._health_checks: Dict[str, Callable[[], ComponentHealth]] = {}
        self._component_health: Dict[str, ComponentHealth] = {}
        
        # Health check configuration
        self._check_intervals: Dict[str, int] = {}  # component -> interval in seconds
        self._last_check_times: Dict[str, datetime] = {}
        
        # Background health checking
        self._background_checking = False
        self._background_thread: Optional[threading.Thread] = None
        self._stop_background = threading.Event()
        
        # Register built-in health checks
        self._register_builtin_checks()
    
    def _register_builtin_checks(self):
        """Register built-in health checks."""
        
        # System resources health check
        self.register_health_check(
            "system_resources",
            self._check_system_resources,
            interval=30
        )
        
        # Disk space health check
        self.register_health_check(
            "disk_space",
            self._check_disk_space,
            interval=60
        )
        
        # Memory usage health check
        self.register_health_check(
            "memory_usage",
            self._check_memory_usage,
            interval=30
        )
    
    def register_health_check(
        self,
        component_name: str,
        check_function: Callable[[], ComponentHealth],
        interval: int = 60
    ):
        """Register a health check for a component."""
        with self._lock:
            self._health_checks[component_name] = check_function
            self._check_intervals[component_name] = interval
            self.logger.info(f"Registered health check for {component_name}")
    
    def unregister_health_check(self, component_name: str):
        """Unregister a health check."""
        with self._lock:
            self._health_checks.pop(component_name, None)
            self._check_intervals.pop(component_name, None)
            self._last_check_times.pop(component_name, None)
            self._component_health.pop(component_name, None)
            self.logger.info(f"Unregistered health check for {component_name}")
    
    def check_component_health(self, component_name: str) -> ComponentHealth:
        """Check health of a specific component."""
        if component_name not in self._health_checks:
            return ComponentHealth(
                component_name=component_name,
                status=HealthStatus.UNKNOWN,
                message="No health check registered",
                last_check=datetime.now(),
                check_duration=0.0
            )
        
        start_time = time.time()
        try:
            health_check = self._health_checks[component_name]
            health = health_check()
            health.check_duration = time.time() - start_time
            
            # Update metrics
            self.metrics.record_timer(f"health_check.{component_name}.duration", health.check_duration)
            self.metrics.increment_counter(f"health_check.{component_name}.total")
            
            if health.status == HealthStatus.HEALTHY:
                self.metrics.increment_counter(f"health_check.{component_name}.healthy")
            else:
                self.metrics.increment_counter(f"health_check.{component_name}.unhealthy")
            
            # Store result
            with self._lock:
                self._component_health[component_name] = health
                self._last_check_times[component_name] = datetime.now()
            
            return health
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Health check failed: {str(e)}"
            
            self.logger.error(f"Health check failed for {component_name}", 
                            component=component_name, 
                            error=str(e),
                            traceback=traceback.format_exc())
            
            health = ComponentHealth(
                component_name=component_name,
                status=HealthStatus.CRITICAL,
                message=error_msg,
                last_check=datetime.now(),
                check_duration=duration,
                error=str(e)
            )
            
            # Update metrics
            self.metrics.record_timer(f"health_check.{component_name}.duration", duration)
            self.metrics.increment_counter(f"health_check.{component_name}.total")
            self.metrics.increment_counter(f"health_check.{component_name}.errors")
            
            with self._lock:
                self._component_health[component_name] = health
                self._last_check_times[component_name] = datetime.now()
            
            return health
    
    def check_all_components(self) -> SystemHealth:
        """Check health of all registered components."""
        component_healths = {}
        
        for component_name in self._health_checks:
            component_healths[component_name] = self.check_component_health(component_name)
        
        return self._calculate_system_health(component_healths)
    
    def get_system_health(self, force_check: bool = False) -> SystemHealth:
        """Get current system health status."""
        if force_check:
            return self.check_all_components()
        
        # Use cached results if available and recent
        with self._lock:
            component_healths = {}
            current_time = datetime.now()
            
            for component_name in self._health_checks:
                # Check if we have recent health data
                last_check = self._last_check_times.get(component_name)
                interval = self._check_intervals.get(component_name, 60)
                
                if (last_check is None or 
                    current_time - last_check > timedelta(seconds=interval * 2)):
                    # Need fresh check
                    component_healths[component_name] = self.check_component_health(component_name)
                else:
                    # Use cached result
                    component_healths[component_name] = self._component_health.get(
                        component_name,
                        ComponentHealth(
                            component_name=component_name,
                            status=HealthStatus.UNKNOWN,
                            message="No recent health data",
                            last_check=current_time,
                            check_duration=0.0
                        )
                    )
        
        return self._calculate_system_health(component_healths)
    
    def _calculate_system_health(self, component_healths: Dict[str, ComponentHealth]) -> SystemHealth:
        """Calculate overall system health from component healths."""
        if not component_healths:
            return SystemHealth(
                overall_status=HealthStatus.UNKNOWN,
                components={},
                check_timestamp=datetime.now(),
                total_components=0,
                healthy_components=0,
                warning_components=0,
                critical_components=0,
                unknown_components=0
            )
        
        # Count components by status
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.WARNING: 0,
            HealthStatus.CRITICAL: 0,
            HealthStatus.UNKNOWN: 0
        }
        
        for health in component_healths.values():
            status_counts[health.status] += 1
        
        # Determine overall status
        if status_counts[HealthStatus.CRITICAL] > 0:
            overall_status = HealthStatus.CRITICAL
        elif status_counts[HealthStatus.WARNING] > 0:
            overall_status = HealthStatus.WARNING
        elif status_counts[HealthStatus.UNKNOWN] > 0:
            overall_status = HealthStatus.WARNING  # Treat unknown as warning
        else:
            overall_status = HealthStatus.HEALTHY
        
        return SystemHealth(
            overall_status=overall_status,
            components=component_healths,
            check_timestamp=datetime.now(),
            total_components=len(component_healths),
            healthy_components=status_counts[HealthStatus.HEALTHY],
            warning_components=status_counts[HealthStatus.WARNING],
            critical_components=status_counts[HealthStatus.CRITICAL],
            unknown_components=status_counts[HealthStatus.UNKNOWN]
        )
    
    def start_background_checking(self, check_interval: int = 30):
        """Start background health checking."""
        if self._background_checking:
            self.logger.warning("Background health checking already running")
            return
        
        self._background_checking = True
        self._stop_background.clear()
        
        def background_check_loop():
            self.logger.info("Started background health checking")
            
            while not self._stop_background.wait(check_interval):
                try:
                    current_time = datetime.now()
                    
                    # Check components that need checking
                    for component_name in list(self._health_checks.keys()):
                        last_check = self._last_check_times.get(component_name)
                        interval = self._check_intervals.get(component_name, 60)
                        
                        if (last_check is None or 
                            current_time - last_check >= timedelta(seconds=interval)):
                            self.check_component_health(component_name)
                
                except Exception as e:
                    self.logger.error(f"Error in background health checking: {e}")
            
            self.logger.info("Stopped background health checking")
        
        self._background_thread = threading.Thread(target=background_check_loop, daemon=True)
        self._background_thread.start()
    
    def stop_background_checking(self):
        """Stop background health checking."""
        if not self._background_checking:
            return
        
        self._background_checking = False
        self._stop_background.set()
        
        if self._background_thread:
            self._background_thread.join(timeout=5)
            self._background_thread = None
    
    def _check_system_resources(self) -> ComponentHealth:
        """Check system resource health."""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Load average (if available)
            load_avg = None
            try:
                import os
                if hasattr(os, 'getloadavg'):
                    load_avg = os.getloadavg()[0]  # 1-minute load average
            except:
                pass
            
            details = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_available_gb': memory.available / (1024**3),
                'load_average': load_avg
            }
            
            # Determine status
            if cpu_percent > 90 or memory_percent > 95:
                status = HealthStatus.CRITICAL
                message = f"High resource usage: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%"
            elif cpu_percent > 80 or memory_percent > 85:
                status = HealthStatus.WARNING
                message = f"Elevated resource usage: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Resource usage normal: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%"
            
            return ComponentHealth(
                component_name="system_resources",
                status=status,
                message=message,
                last_check=datetime.now(),
                check_duration=0.0,
                details=details
            )
            
        except Exception as e:
            return ComponentHealth(
                component_name="system_resources",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check system resources: {e}",
                last_check=datetime.now(),
                check_duration=0.0,
                error=str(e)
            )
    
    def _check_disk_space(self) -> ComponentHealth:
        """Check disk space health."""
        try:
            import psutil
            
            # Check root disk
            disk_usage = psutil.disk_usage('/')
            used_percent = (disk_usage.used / disk_usage.total) * 100
            free_gb = disk_usage.free / (1024**3)
            
            details = {
                'used_percent': used_percent,
                'free_gb': free_gb,
                'total_gb': disk_usage.total / (1024**3)
            }
            
            # Determine status
            if used_percent > 95 or free_gb < 1:
                status = HealthStatus.CRITICAL
                message = f"Disk space critical: {used_percent:.1f}% used, {free_gb:.1f}GB free"
            elif used_percent > 85 or free_gb < 5:
                status = HealthStatus.WARNING
                message = f"Disk space low: {used_percent:.1f}% used, {free_gb:.1f}GB free"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk space OK: {used_percent:.1f}% used, {free_gb:.1f}GB free"
            
            return ComponentHealth(
                component_name="disk_space",
                status=status,
                message=message,
                last_check=datetime.now(),
                check_duration=0.0,
                details=details
            )
            
        except Exception as e:
            return ComponentHealth(
                component_name="disk_space",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check disk space: {e}",
                last_check=datetime.now(),
                check_duration=0.0,
                error=str(e)
            )
    
    def _check_memory_usage(self) -> ComponentHealth:
        """Check memory usage health."""
        try:
            import psutil
            import os
            
            # System memory
            system_memory = psutil.virtual_memory()
            
            # Process memory
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info()
            
            details = {
                'system_memory_percent': system_memory.percent,
                'system_memory_available_gb': system_memory.available / (1024**3),
                'process_memory_rss_mb': process_memory.rss / (1024**2),
                'process_memory_vms_mb': process_memory.vms / (1024**2)
            }
            
            # Determine status based on system memory
            if system_memory.percent > 95:
                status = HealthStatus.CRITICAL
                message = f"System memory critical: {system_memory.percent:.1f}% used"
            elif system_memory.percent > 85:
                status = HealthStatus.WARNING
                message = f"System memory high: {system_memory.percent:.1f}% used"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {system_memory.percent:.1f}% used"
            
            return ComponentHealth(
                component_name="memory_usage",
                status=status,
                message=message,
                last_check=datetime.now(),
                check_duration=0.0,
                details=details
            )
            
        except Exception as e:
            return ComponentHealth(
                component_name="memory_usage",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check memory usage: {e}",
                last_check=datetime.now(),
                check_duration=0.0,
                error=str(e)
            )
    
    def create_model_health_check(self, model_name: str, model_loader_func: Callable) -> Callable[[], ComponentHealth]:
        """Create a health check function for a model."""
        def check_model_health() -> ComponentHealth:
            try:
                start_time = time.time()
                
                # Try to load/access the model
                model = model_loader_func()
                
                load_time = time.time() - start_time
                
                details = {
                    'model_name': model_name,
                    'load_time': load_time,
                    'model_loaded': model is not None
                }
                
                if model is None:
                    return ComponentHealth(
                        component_name=f"model_{model_name}",
                        status=HealthStatus.CRITICAL,
                        message=f"Model {model_name} failed to load",
                        last_check=datetime.now(),
                        check_duration=load_time,
                        details=details
                    )
                
                # Model loaded successfully
                if load_time > 30:  # Slow loading
                    status = HealthStatus.WARNING
                    message = f"Model {model_name} loaded slowly ({load_time:.1f}s)"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Model {model_name} loaded successfully ({load_time:.1f}s)"
                
                return ComponentHealth(
                    component_name=f"model_{model_name}",
                    status=status,
                    message=message,
                    last_check=datetime.now(),
                    check_duration=load_time,
                    details=details
                )
                
            except Exception as e:
                return ComponentHealth(
                    component_name=f"model_{model_name}",
                    status=HealthStatus.CRITICAL,
                    message=f"Model {model_name} health check failed: {e}",
                    last_check=datetime.now(),
                    check_duration=0.0,
                    error=str(e)
                )
        
        return check_model_health
    
    def create_storage_health_check(self, storage_name: str, storage_path: str) -> Callable[[], ComponentHealth]:
        """Create a health check function for storage."""
        def check_storage_health() -> ComponentHealth:
            try:
                path = Path(storage_path)
                
                details = {
                    'storage_name': storage_name,
                    'storage_path': storage_path,
                    'path_exists': path.exists(),
                    'is_directory': path.is_dir() if path.exists() else False,
                    'is_writable': os.access(path.parent if not path.exists() else path, os.W_OK)
                }
                
                if not path.exists():
                    # Try to create the path
                    try:
                        path.mkdir(parents=True, exist_ok=True)
                        details['path_exists'] = True
                        details['is_directory'] = True
                    except Exception as create_error:
                        return ComponentHealth(
                            component_name=f"storage_{storage_name}",
                            status=HealthStatus.CRITICAL,
                            message=f"Storage path {storage_path} does not exist and cannot be created",
                            last_check=datetime.now(),
                            check_duration=0.0,
                            details=details,
                            error=str(create_error)
                        )
                
                if not details['is_writable']:
                    return ComponentHealth(
                        component_name=f"storage_{storage_name}",
                        status=HealthStatus.CRITICAL,
                        message=f"Storage path {storage_path} is not writable",
                        last_check=datetime.now(),
                        check_duration=0.0,
                        details=details
                    )
                
                return ComponentHealth(
                    component_name=f"storage_{storage_name}",
                    status=HealthStatus.HEALTHY,
                    message=f"Storage {storage_name} is accessible",
                    last_check=datetime.now(),
                    check_duration=0.0,
                    details=details
                )
                
            except Exception as e:
                return ComponentHealth(
                    component_name=f"storage_{storage_name}",
                    status=HealthStatus.CRITICAL,
                    message=f"Storage {storage_name} health check failed: {e}",
                    last_check=datetime.now(),
                    check_duration=0.0,
                    error=str(e)
                )
        
        return check_storage_health
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of system health."""
        system_health = self.get_system_health()
        
        return {
            'overall_status': system_health.overall_status.value,
            'check_timestamp': system_health.check_timestamp.isoformat(),
            'component_summary': {
                'total': system_health.total_components,
                'healthy': system_health.healthy_components,
                'warning': system_health.warning_components,
                'critical': system_health.critical_components,
                'unknown': system_health.unknown_components
            },
            'components': {
                name: {
                    'status': health.status.value,
                    'message': health.message,
                    'last_check': health.last_check.isoformat(),
                    'check_duration': health.check_duration,
                    'details': health.details,
                    'error': health.error
                }
                for name, health in system_health.components.items()
            }
        }


# Global health checker instance
_health_checker = HealthChecker()


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    return _health_checker


def check_system_health(force_check: bool = False) -> SystemHealth:
    """Check overall system health."""
    return _health_checker.get_system_health(force_check)