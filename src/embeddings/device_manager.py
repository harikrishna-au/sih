"""
Device management utilities for embedding models.

Handles device detection and optimization for embedding generation
across different hardware configurations (CPU, CUDA, MPS).
"""

import logging
from typing import Optional

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages device selection and optimization for embedding models."""
    
    @staticmethod
    def determine_device(preferred_device: str = "auto") -> str:
        """
        Determine the best available device for embedding generation.
        
        Args:
            preferred_device: Preferred device ('auto', 'cuda', 'mps', 'cpu')
            
        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        if preferred_device != "auto":
            return preferred_device
        
        if torch is None:
            logger.warning("PyTorch not available, using CPU")
            return "cpu"
        
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"CUDA available, using GPU: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Apple Metal Performance Shaders available, using MPS")
        else:
            device = "cpu"
            logger.info("Using CPU for embedding generation")
        
        return device
    
    @staticmethod
    def get_device_info(device: str) -> dict:
        """
        Get information about the specified device.
        
        Args:
            device: Device string
            
        Returns:
            Dictionary with device information
        """
        info = {"device": device, "available": True}
        
        if torch is None:
            info["available"] = False
            info["reason"] = "PyTorch not installed"
            return info
        
        if device == "cuda":
            if torch.cuda.is_available():
                info["device_count"] = torch.cuda.device_count()
                info["device_name"] = torch.cuda.get_device_name()
                info["memory_total"] = torch.cuda.get_device_properties(0).total_memory
            else:
                info["available"] = False
                info["reason"] = "CUDA not available"
        
        elif device == "mps":
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                info["backend"] = "Metal Performance Shaders"
            else:
                info["available"] = False
                info["reason"] = "MPS not available"
        
        elif device == "cpu":
            info["cpu_count"] = torch.get_num_threads() if torch else "unknown"
        
        return info
    
    @staticmethod
    def optimize_for_device(device: str) -> dict:
        """
        Get optimization settings for the specified device.
        
        Args:
            device: Device string
            
        Returns:
            Dictionary with optimization settings
        """
        settings = {
            "batch_size": 32,
            "num_workers": 4,
            "pin_memory": False
        }
        
        if device == "cuda":
            settings.update({
                "batch_size": 64,
                "num_workers": 8,
                "pin_memory": True
            })
        elif device == "mps":
            settings.update({
                "batch_size": 48,
                "num_workers": 6,
                "pin_memory": True
            })
        elif device == "cpu":
            settings.update({
                "batch_size": 16,
                "num_workers": 2,
                "pin_memory": False
            })
        
        return settings