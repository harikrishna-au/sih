"""
Image processor for extracting content and metadata from images.

Uses PIL for image processing and OCR for text extraction.
"""

import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

try:
    from PIL import Image, ExifTags
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from .base import DocumentProcessor, ProcessingError, ContentExtractionError, FileValidationError
from ..models import (
    DocumentMetadata, ContentChunk, ValidationResult, ContentType, SourceLocation
)
from ..config import ProcessingConfig

logger = logging.getLogger(__name__)


class ImageProcessor(DocumentProcessor):
    """
    Image processor that extracts metadata and optional OCR text.
    
    Features:
    - Image metadata extraction (dimensions, format, EXIF data)
    - Basic image description generation
    - File validation and error handling
    """
    
    def __init__(self, config: ProcessingConfig):
        super().__init__(config)
        self.supported_formats = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']
        
        if not PIL_AVAILABLE:
            logger.warning("PIL not available. Image processing will be limited.")
    
    def validate_file(self, file_path: str) -> ValidationResult:
        """
        Validate that the file is a readable image.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            ValidationResult with validation status and any errors
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                return ValidationResult(
                    is_valid=False,
                    error_message=f"File not found: {file_path}"
                )
            
            # Check file extension
            file_extension = path.suffix.lower().lstrip('.')
            if file_extension not in self.supported_formats:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Unsupported image format: {file_extension}",
                    file_format=file_extension
                )
            
            # Check file size
            if self._is_file_too_large(file_path):
                return ValidationResult(
                    is_valid=False,
                    error_message=f"File too large. Maximum size: {self.config.max_file_size_mb}MB",
                    file_size=self._get_file_size(file_path)
                )
            
            # Try to open the image
            if PIL_AVAILABLE:
                try:
                    with Image.open(file_path) as img:
                        img.verify()  # Verify it's a valid image
                except Exception as e:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Invalid image file: {str(e)}",
                        file_format=file_extension
                    )
            
            return ValidationResult(
                is_valid=True,
                file_format=file_extension,
                file_size=self._get_file_size(file_path)
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Validation error: {str(e)}"
            )
    
    def extract_content(self, file_path: str) -> str:
        """
        Extract descriptive content from the image.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Descriptive text about the image
            
        Raises:
            ContentExtractionError: If content extraction fails
        """
        try:
            if not PIL_AVAILABLE:
                # Fallback description without PIL
                path = Path(file_path)
                return f"Image file: {path.name}"
            
            with Image.open(file_path) as img:
                # Get basic image information
                width, height = img.size
                format_name = img.format or "Unknown"
                mode = img.mode
                
                # Create descriptive content
                content_parts = [
                    f"Image file: {Path(file_path).name}",
                    f"Format: {format_name}",
                    f"Dimensions: {width}x{height} pixels",
                    f"Color mode: {mode}"
                ]
                
                # Add EXIF data if available
                if hasattr(img, '_getexif') and img._getexif():
                    exif_data = img._getexif()
                    if exif_data:
                        # Extract common EXIF tags
                        for tag_id, value in exif_data.items():
                            tag = ExifTags.TAGS.get(tag_id, tag_id)
                            if tag in ['DateTime', 'Make', 'Model', 'Software']:
                                content_parts.append(f"{tag}: {value}")
                
                return "\n".join(content_parts)
                
        except FileNotFoundError:
            raise ContentExtractionError(
                f"Image file not found: {file_path}",
                file_path=file_path
            )
        except Exception as e:
            raise ContentExtractionError(
                f"Failed to extract content from image: {str(e)}",
                file_path=file_path,
                cause=e
            )
    
    def extract_metadata(self, file_path: str) -> DocumentMetadata:
        """
        Extract metadata from the image file.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            DocumentMetadata with extracted information
        """
        try:
            path = Path(file_path)
            file_size = self._get_file_size(file_path)
            
            if not PIL_AVAILABLE:
                return DocumentMetadata(
                    title=path.stem,
                    file_size=file_size,
                    confidence_score=0.5
                )
            
            with Image.open(file_path) as img:
                width, height = img.size
                format_name = img.format
                
                # Extract creation date from EXIF if available
                creation_date = None
                if hasattr(img, '_getexif') and img._getexif():
                    exif_data = img._getexif()
                    if exif_data and 'DateTime' in [ExifTags.TAGS.get(k) for k in exif_data.keys()]:
                        for tag_id, value in exif_data.items():
                            if ExifTags.TAGS.get(tag_id) == 'DateTime':
                                try:
                                    # Convert EXIF datetime to ISO format
                                    dt = datetime.strptime(str(value), '%Y:%m:%d %H:%M:%S')
                                    creation_date = dt.strftime('%Y-%m-%d')
                                except ValueError:
                                    pass
                                break
                
                return DocumentMetadata(
                    title=path.stem,
                    file_size=file_size,
                    dimensions=(width, height),
                    format_version=format_name,
                    creation_date=creation_date,
                    confidence_score=1.0
                )
                
        except Exception as e:
            logger.warning(f"Error extracting image metadata: {e}")
            return DocumentMetadata(
                title=Path(file_path).stem,
                file_size=self._get_file_size(file_path),
                confidence_score=0.5
            )
    
    def chunk_content(self, content: str, document_id: str, file_path: str) -> List[ContentChunk]:
        """
        Create a single chunk for the image content.
        
        Args:
            content: Processed content to chunk
            document_id: Unique identifier for the document
            file_path: Original file path for source location
            
        Returns:
            List containing a single ContentChunk for the image
        """
        chunk_id = self._create_chunk_id(document_id, 0)
        
        source_location = SourceLocation(
            file_path=file_path
        )
        
        chunk = ContentChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            content=content,
            content_type=ContentType.IMAGE,
            source_location=source_location,
            metadata={
                'chunk_index': 0,
                'character_count': len(content),
                'word_count': len(content.split())
            },
            confidence_score=1.0
        )
        
        return [chunk]
    
    def get_content_type(self) -> ContentType:
        """Get the content type handled by this processor."""
        return ContentType.IMAGE
    
    def process_content(self, raw_content: str) -> str:
        """
        Process raw image content.
        
        Args:
            raw_content: Raw extracted content
            
        Returns:
            Processed content ready for chunking
        """
        return raw_content.strip()