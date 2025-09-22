"""
Document router for file validation and format detection.

The DocumentRouter identifies file types and routes them to appropriate
processors while providing comprehensive file validation.
"""

import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Type
import magic

from .base import DocumentProcessor, UnsupportedFormatError, FileValidationError
from ..models import ValidationResult, ContentType
from ..config import ProcessingConfig


class DocumentRouter:
    """
    Routes documents to appropriate processors based on file format detection.
    
    Provides file validation, format detection, and processor routing
    for all supported document types.
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self._processors: Dict[str, Type[DocumentProcessor]] = {}
        self._format_mappings = self._build_format_mappings()
        
        # Initialize python-magic for MIME type detection
        try:
            self._magic = magic.Magic(mime=True)
        except Exception:
            # Fallback if python-magic is not available
            self._magic = None
    
    def register_processor(self, processor_class: Type[DocumentProcessor], formats: List[str]) -> None:
        """
        Register a processor for specific file formats.
        
        Args:
            processor_class: The processor class to register
            formats: List of file extensions (without dots) this processor handles
        """
        for fmt in formats:
            self._processors[fmt.lower()] = processor_class
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of all supported file formats.
        
        Returns:
            List of supported file extensions (without dots)
        """
        return list(self._processors.keys())
    
    def detect_file_format(self, file_path: str) -> Optional[str]:
        """
        Detect file format using multiple methods.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected file format (extension without dot) or None if unknown
        """
        path = Path(file_path)
        
        # Method 1: File extension (check against known formats)
        extension = path.suffix.lower().lstrip('.')
        known_formats = ['pdf', 'docx', 'doc', 'txt', 'png', 'jpg', 'jpeg', 'mp3', 'wav', 'm4a']
        if extension in known_formats:
            return extension
        
        # Method 2: MIME type detection
        mime_format = self._detect_by_mime_type(file_path)
        if mime_format and mime_format in known_formats:
            return mime_format
        
        # Method 3: File signature detection
        signature_format = self._detect_by_signature(file_path)
        if signature_format and signature_format in known_formats:
            return signature_format
        
        return None
    
    def validate_file(self, file_path: str) -> ValidationResult:
        """
        Comprehensive file validation.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            ValidationResult with validation status and details
        """
        path = Path(file_path)
        warnings = []
        
        # Check if file exists
        if not path.exists():
            return ValidationResult(
                is_valid=False,
                error_message=f"File does not exist: {file_path}"
            )
        
        # Check if it's a file (not directory)
        if not path.is_file():
            return ValidationResult(
                is_valid=False,
                error_message=f"Path is not a file: {file_path}"
            )
        
        # Check file size
        file_size = path.stat().st_size
        max_size_bytes = self.config.max_file_size_mb * 1024 * 1024
        
        if file_size == 0:
            return ValidationResult(
                is_valid=False,
                error_message=f"File is empty: {file_path}",
                file_size=file_size
            )
        
        if file_size > max_size_bytes:
            return ValidationResult(
                is_valid=False,
                error_message=f"File too large: {file_size} bytes (max: {max_size_bytes} bytes)",
                file_size=file_size
            )
        
        # Detect file format
        detected_format = self.detect_file_format(file_path)
        
        if not detected_format:
            return ValidationResult(
                is_valid=False,
                error_message=f"Unsupported file format: {file_path}",
                file_size=file_size
            )
        
        # Check if format is in supported list
        if detected_format not in self.config.supported_formats:
            return ValidationResult(
                is_valid=False,
                error_message=f"File format '{detected_format}' is not enabled in configuration",
                file_format=detected_format,
                file_size=file_size
            )
        
        # Check file permissions
        if not path.stat().st_mode & 0o444:  # Check read permission
            warnings.append("File may not be readable due to permissions")
        
        # Additional format-specific validation
        format_validation = self._validate_format_specific(file_path, detected_format)
        if not format_validation.is_valid:
            return format_validation
        
        warnings.extend(format_validation.warnings)
        
        return ValidationResult(
            is_valid=True,
            file_format=detected_format,
            file_size=file_size,
            warnings=warnings
        )
    
    def route_document(self, file_path: str) -> DocumentProcessor:
        """
        Route document to appropriate processor.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Instantiated processor for the file type
            
        Raises:
            UnsupportedFormatError: If file format is not supported
            FileValidationError: If file validation fails
        """
        # Validate file first
        validation = self.validate_file(file_path)
        if not validation.is_valid:
            raise FileValidationError(validation.error_message, file_path)
        
        # Get processor class
        processor_class = self._processors.get(validation.file_format)
        if not processor_class:
            raise UnsupportedFormatError(
                f"No processor available for format: {validation.file_format}",
                file_path
            )
        
        # Instantiate and return processor
        return processor_class(self.config)
    
    def _build_format_mappings(self) -> Dict[str, str]:
        """Build MIME type to format mappings."""
        return {
            'application/pdf': 'pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
            'application/msword': 'doc',
            'text/plain': 'txt',
            'image/png': 'png',
            'image/jpeg': 'jpg',
            'image/jpg': 'jpg',
            'audio/mpeg': 'mp3',
            'audio/wav': 'wav',
            'audio/wave': 'wav',
            'audio/x-wav': 'wav',
            'audio/mp4': 'm4a',
            'audio/x-m4a': 'm4a',
        }
    
    def _detect_by_mime_type(self, file_path: str) -> Optional[str]:
        """Detect format using MIME type."""
        try:
            # Try python-magic first (more accurate)
            if self._magic:
                mime_type = self._magic.from_file(file_path)
            else:
                # Fallback to mimetypes module
                mime_type, _ = mimetypes.guess_type(file_path)
            
            if mime_type:
                return self._format_mappings.get(mime_type)
        except Exception:
            pass
        
        return None
    
    def _detect_by_signature(self, file_path: str) -> Optional[str]:
        """Detect format using file signatures (magic numbers)."""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(16)
            
            # PDF signature
            if header.startswith(b'%PDF'):
                return 'pdf'
            
            # PNG signature
            if header.startswith(b'\x89PNG\r\n\x1a\n'):
                return 'png'
            
            # JPEG signatures
            if header.startswith(b'\xff\xd8\xff'):
                return 'jpg'
            
            # DOCX/ZIP signature (DOCX is a ZIP file)
            if header.startswith(b'PK\x03\x04') or header.startswith(b'PK\x05\x06'):
                # Need to check if it's actually a DOCX
                return self._check_docx_signature(file_path)
            
            # MP3 signature
            if header.startswith(b'ID3') or header.startswith(b'\xff\xfb'):
                return 'mp3'
            
            # WAV signature
            if header.startswith(b'RIFF') and b'WAVE' in header:
                return 'wav'
            
        except Exception:
            pass
        
        return None
    
    def _check_docx_signature(self, file_path: str) -> Optional[str]:
        """Check if ZIP file is actually a DOCX."""
        try:
            import zipfile
            with zipfile.ZipFile(file_path, 'r') as zip_file:
                # DOCX files contain specific files
                if '[Content_Types].xml' in zip_file.namelist():
                    return 'docx'
        except Exception:
            pass
        
        return None
    
    def _validate_format_specific(self, file_path: str, file_format: str) -> ValidationResult:
        """Perform format-specific validation."""
        warnings = []
        
        try:
            if file_format == 'pdf':
                return self._validate_pdf(file_path)
            elif file_format == 'docx':
                return self._validate_docx(file_path)
            elif file_format in ['png', 'jpg', 'jpeg']:
                return self._validate_image(file_path)
            elif file_format in ['mp3', 'wav', 'm4a']:
                return self._validate_audio(file_path)
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Format validation failed: {str(e)}"
            )
        
        return ValidationResult(is_valid=True, warnings=warnings)
    
    def _validate_pdf(self, file_path: str) -> ValidationResult:
        """Validate PDF file structure."""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(8)
                if not header.startswith(b'%PDF-'):
                    return ValidationResult(
                        is_valid=False,
                        error_message="Invalid PDF header"
                    )
                
                # Check PDF version
                version = header[5:8].decode('ascii', errors='ignore')
                if version not in ['1.0', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '2.0']:
                    return ValidationResult(
                        is_valid=True,
                        warnings=[f"Unknown PDF version: {version}"]
                    )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"PDF validation error: {str(e)}"
            )
        
        return ValidationResult(is_valid=True)
    
    def _validate_docx(self, file_path: str) -> ValidationResult:
        """Validate DOCX file structure."""
        try:
            import zipfile
            with zipfile.ZipFile(file_path, 'r') as zip_file:
                required_files = ['[Content_Types].xml', '_rels/.rels']
                missing_files = [f for f in required_files if f not in zip_file.namelist()]
                
                if missing_files:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Invalid DOCX structure, missing: {missing_files}"
                    )
        except zipfile.BadZipFile:
            return ValidationResult(
                is_valid=False,
                error_message="File is not a valid ZIP/DOCX archive"
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"DOCX validation error: {str(e)}"
            )
        
        return ValidationResult(is_valid=True)
    
    def _validate_image(self, file_path: str) -> ValidationResult:
        """Validate image file."""
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                # Check if image can be loaded
                img.verify()
                
                # Check reasonable dimensions
                if img.size[0] > 10000 or img.size[1] > 10000:
                    return ValidationResult(
                        is_valid=True,
                        warnings=["Image dimensions are very large, processing may be slow"]
                    )
        except ImportError:
            # PIL not available, basic validation only
            pass
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Image validation error: {str(e)}"
            )
        
        return ValidationResult(is_valid=True)
    
    def _validate_audio(self, file_path: str) -> ValidationResult:
        """Validate audio file."""
        warnings = []
        
        try:
            # Basic file size check for audio
            file_size = Path(file_path).stat().st_size
            
            # Warn for very large audio files (>100MB)
            if file_size > 100 * 1024 * 1024:
                warnings.append("Large audio file, processing may take significant time")
            
            # Try to get basic audio info if librosa is available
            try:
                import librosa
                duration = librosa.get_duration(path=file_path)
                
                # Warn for very long audio files (>1 hour)
                if duration > 3600:
                    warnings.append("Audio file is longer than 1 hour, consider splitting")
                    
            except ImportError:
                # librosa not available, skip duration check
                pass
            except Exception:
                # Audio file might not be valid audio or librosa can't read it
                # This is not necessarily an error, just skip duration check
                pass
                
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Audio validation error: {str(e)}"
            )
        
        return ValidationResult(is_valid=True, warnings=warnings)