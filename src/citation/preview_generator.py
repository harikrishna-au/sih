"""
Preview data generation for citations.

This module implements preview generation for different content types including
PDF snippets, image thumbnails, and audio clips with caching and optimization.
"""

import logging
import hashlib
import io
import base64
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF for PDF handling
import wave
import numpy as np
from ..models import ContentType, SourceLocation, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class PreviewConfig:
    """Configuration for preview generation."""
    # Image settings
    image_thumbnail_size: Tuple[int, int] = (200, 200)
    image_quality: int = 85
    
    # PDF settings
    pdf_snippet_width: int = 400
    pdf_snippet_height: int = 300
    pdf_dpi: int = 150
    
    # Audio settings
    audio_clip_duration: float = 10.0  # seconds
    audio_sample_rate: int = 16000
    audio_format: str = 'wav'
    
    # General settings
    max_preview_size_kb: int = 100
    cache_enabled: bool = True
    cache_max_size: int = 1000


class PreviewCache:
    """Simple in-memory cache for preview data."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, bytes] = {}
        self._access_order: List[str] = []
    
    def get(self, key: str) -> Optional[bytes]:
        """Get cached preview data."""
        if key in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    
    def put(self, key: str, data: bytes) -> None:
        """Store preview data in cache."""
        if key in self._cache:
            # Update existing entry
            self._cache[key] = data
            self._access_order.remove(key)
            self._access_order.append(key)
        else:
            # Add new entry
            if len(self._cache) >= self.max_size:
                # Remove least recently used
                oldest_key = self._access_order.pop(0)
                del self._cache[oldest_key]
            
            self._cache[key] = data
            self._access_order.append(key)
    
    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._access_order.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


class PreviewGenerator:
    """
    Generates preview data for different content types with caching and optimization.
    
    Supports:
    - PDF snippets: Rendered page regions as images
    - Image thumbnails: Resized images with optional region highlighting
    - Audio clips: Short audio segments around specified timestamps
    """
    
    def __init__(self, config: Optional[PreviewConfig] = None):
        """Initialize the preview generator."""
        self.config = config or PreviewConfig()
        self.cache = PreviewCache(self.config.cache_max_size) if self.config.cache_enabled else None
        
    def generate_preview(
        self, 
        retrieval_result: RetrievalResult,
        force_regenerate: bool = False
    ) -> Optional[bytes]:
        """
        Generate preview data for a retrieval result.
        
        Args:
            retrieval_result: The retrieval result to generate preview for
            force_regenerate: Whether to bypass cache and regenerate
            
        Returns:
            Preview data as bytes, or None if generation fails
        """
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(retrieval_result)
            
            # Check cache first
            if not force_regenerate and self.cache:
                cached_data = self.cache.get(cache_key)
                if cached_data:
                    logger.debug(f"Using cached preview for {cache_key}")
                    return cached_data
            
            # Generate preview based on content type
            preview_data = None
            
            if retrieval_result.content_type == ContentType.PDF:
                preview_data = self._generate_pdf_preview(retrieval_result)
            elif retrieval_result.content_type == ContentType.IMAGE:
                preview_data = self._generate_image_preview(retrieval_result)
            elif retrieval_result.content_type == ContentType.AUDIO:
                preview_data = self._generate_audio_preview(retrieval_result)
            else:
                logger.warning(f"Preview generation not supported for content type: {retrieval_result.content_type}")
                return None
            
            # Optimize size if needed
            if preview_data:
                preview_data = self._optimize_preview_size(preview_data, retrieval_result.content_type)
                
                # Cache the result
                if self.cache:
                    self.cache.put(cache_key, preview_data)
                    logger.debug(f"Cached preview for {cache_key}")
            
            return preview_data
            
        except Exception as e:
            logger.error(f"Failed to generate preview: {e}")
            return None
    
    def _generate_pdf_preview(self, result: RetrievalResult) -> Optional[bytes]:
        """Generate a preview snippet from a PDF page."""
        try:
            file_path = result.source_location.file_path
            page_number = result.source_location.page_number or 1
            
            if not Path(file_path).exists():
                logger.error(f"PDF file not found: {file_path}")
                return None
            
            # Open PDF document
            doc = fitz.open(file_path)
            
            if page_number > len(doc):
                logger.error(f"Page {page_number} not found in PDF {file_path}")
                return None
            
            # Get the page (0-indexed)
            page = doc[page_number - 1]
            
            # Create transformation matrix for rendering
            mat = fitz.Matrix(self.config.pdf_dpi / 72, self.config.pdf_dpi / 72)
            
            # Render page to pixmap
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Resize to target dimensions while maintaining aspect ratio
            img.thumbnail((self.config.pdf_snippet_width, self.config.pdf_snippet_height), Image.Resampling.LANCZOS)
            
            # Convert to bytes
            output = io.BytesIO()
            img.save(output, format='PNG', optimize=True)
            
            doc.close()
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Failed to generate PDF preview: {e}")
            return None
    
    def _generate_image_preview(self, result: RetrievalResult) -> Optional[bytes]:
        """Generate a thumbnail preview from an image."""
        try:
            file_path = result.source_location.file_path
            
            if not Path(file_path).exists():
                logger.error(f"Image file not found: {file_path}")
                return None
            
            # Open and process image
            with Image.open(file_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Create thumbnail
                img.thumbnail(self.config.image_thumbnail_size, Image.Resampling.LANCZOS)
                
                # Highlight region if coordinates are provided
                if result.source_location.image_coordinates:
                    img = self._highlight_image_region(img, result.source_location.image_coordinates)
                
                # Convert to bytes
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=self.config.image_quality, optimize=True)
                return output.getvalue()
                
        except Exception as e:
            logger.error(f"Failed to generate image preview: {e}")
            return None
    
    def _generate_audio_preview(self, result: RetrievalResult) -> Optional[bytes]:
        """Generate a short audio clip preview."""
        try:
            file_path = result.source_location.file_path
            timestamp_start = result.source_location.timestamp_start or 0.0
            
            if not Path(file_path).exists():
                logger.error(f"Audio file not found: {file_path}")
                return None
            
            # For now, we'll create a simple waveform visualization
            # In a full implementation, you might extract actual audio segments
            return self._generate_waveform_preview(file_path, timestamp_start)
            
        except Exception as e:
            logger.error(f"Failed to generate audio preview: {e}")
            return None
    
    def _generate_waveform_preview(self, file_path: str, timestamp_start: float) -> Optional[bytes]:
        """Generate a waveform visualization as audio preview."""
        try:
            # Create a simple waveform visualization
            # This is a placeholder - in production you'd use actual audio processing
            width, height = 300, 100
            img = Image.new('RGB', (width, height), 'white')
            draw = ImageDraw.Draw(img)
            
            # Generate mock waveform data
            np.random.seed(int(timestamp_start * 1000) % 2**32)  # Deterministic based on timestamp
            waveform = np.random.normal(0, 0.3, width)
            
            # Draw waveform
            center_y = height // 2
            for i, amplitude in enumerate(waveform):
                y_offset = int(amplitude * center_y)
                draw.line([(i, center_y - y_offset), (i, center_y + y_offset)], fill='blue', width=1)
            
            # Add timestamp marker
            marker_x = width // 2
            draw.line([(marker_x, 0), (marker_x, height)], fill='red', width=2)
            
            # Add timestamp text
            try:
                font = ImageFont.load_default()
                timestamp_text = f"{timestamp_start:.1f}s"
                draw.text((marker_x + 5, 5), timestamp_text, fill='red', font=font)
            except:
                pass  # Font loading might fail in some environments
            
            # Convert to bytes
            output = io.BytesIO()
            img.save(output, format='PNG', optimize=True)
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Failed to generate waveform preview: {e}")
            return None
    
    def _highlight_image_region(self, img: Image.Image, coordinates: Tuple[int, int, int, int]) -> Image.Image:
        """Highlight a specific region in an image."""
        try:
            x, y, width, height = coordinates
            draw = ImageDraw.Draw(img)
            
            # Scale coordinates to thumbnail size
            original_size = img.size
            scale_x = original_size[0] / self.config.image_thumbnail_size[0]
            scale_y = original_size[1] / self.config.image_thumbnail_size[1]
            
            scaled_x = int(x / scale_x)
            scaled_y = int(y / scale_y)
            scaled_width = int(width / scale_x)
            scaled_height = int(height / scale_y)
            
            # Draw highlight rectangle
            draw.rectangle(
                [scaled_x, scaled_y, scaled_x + scaled_width, scaled_y + scaled_height],
                outline='red',
                width=2
            )
            
            return img
            
        except Exception as e:
            logger.error(f"Failed to highlight image region: {e}")
            return img
    
    def _optimize_preview_size(self, data: bytes, content_type: ContentType) -> bytes:
        """Optimize preview data size to stay within limits."""
        max_size_bytes = self.config.max_preview_size_kb * 1024
        
        if len(data) <= max_size_bytes:
            return data
        
        try:
            if content_type in [ContentType.PDF, ContentType.IMAGE]:
                # Re-compress image with lower quality
                img = Image.open(io.BytesIO(data))
                output = io.BytesIO()
                
                # Reduce quality progressively
                for quality in [70, 50, 30]:
                    output.seek(0)
                    output.truncate()
                    img.save(output, format='JPEG', quality=quality, optimize=True)
                    
                    if len(output.getvalue()) <= max_size_bytes:
                        return output.getvalue()
                
                # If still too large, resize the image
                img.thumbnail((150, 150), Image.Resampling.LANCZOS)
                output.seek(0)
                output.truncate()
                img.save(output, format='JPEG', quality=50, optimize=True)
                return output.getvalue()
            
            else:
                # For other types, just truncate if necessary
                logger.warning(f"Preview size optimization not implemented for {content_type}")
                return data[:max_size_bytes]
                
        except Exception as e:
            logger.error(f"Failed to optimize preview size: {e}")
            return data
    
    def _generate_cache_key(self, result: RetrievalResult) -> str:
        """Generate a unique cache key for a retrieval result."""
        key_components = [
            result.source_location.file_path,
            str(result.source_location.page_number or ''),
            str(result.source_location.timestamp_start or ''),
            str(result.source_location.image_coordinates or ''),
            result.content_type.value
        ]
        
        key_string = '|'.join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.cache:
            return {"cache_enabled": False}
        
        return {
            "cache_enabled": True,
            "cache_size": self.cache.size(),
            "max_cache_size": self.config.cache_max_size
        }
    
    def clear_cache(self) -> None:
        """Clear the preview cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Preview cache cleared")