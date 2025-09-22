"""
DOCX file validation utilities.

Provides validation functionality for DOCX files including format checking,
readability verification, and basic structure validation.
"""

import logging
from pathlib import Path
from typing import Optional

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

from ..base import FileValidationError
from ...models import ValidationResult

logger = logging.getLogger(__name__)


class DOCXValidator:
    """Handles validation of DOCX files."""
    
    def __init__(self):
        """Initialize DOCX validator."""
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx is required for DOCX processing. Install with: pip install python-docx>=0.8.11")
    
    def validate_file(self, file_path: str) -> ValidationResult:
        """
        Validate that the file is a readable DOCX document.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            ValidationResult with validation status and details
        """
        try:
            file_path_obj = Path(file_path)
            
            # Check if file exists
            if not file_path_obj.exists():
                return ValidationResult(
                    is_valid=False,
                    error_message=f"File does not exist: {file_path}",
                    file_path=file_path
                )
            
            # Check file extension
            if file_path_obj.suffix.lower() != '.docx':
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Invalid file extension. Expected .docx, got {file_path_obj.suffix}",
                    file_path=file_path
                )
            
            # Check file size
            file_size = file_path_obj.stat().st_size
            if file_size == 0:
                return ValidationResult(
                    is_valid=False,
                    error_message="File is empty",
                    file_path=file_path
                )
            
            # Try to open and read the document
            try:
                doc = Document(file_path)
                
                # Basic structure validation
                validation_issues = self._validate_document_structure(doc)
                
                if validation_issues:
                    return ValidationResult(
                        is_valid=True,  # Still valid but with warnings
                        warnings=validation_issues,
                        file_path=file_path,
                        file_size=file_size
                    )
                
                return ValidationResult(
                    is_valid=True,
                    file_path=file_path,
                    file_size=file_size
                )
                
            except Exception as e:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Cannot read DOCX file: {str(e)}",
                    file_path=file_path
                )
                
        except Exception as e:
            logger.error(f"Validation error for {file_path}: {e}")
            raise FileValidationError(
                f"Failed to validate DOCX file {file_path}: {str(e)}",
                file_path=file_path,
                cause=e
            )
    
    def _validate_document_structure(self, doc) -> list:
        """
        Validate document structure and return any issues.
        
        Args:
            doc: Loaded DOCX document
            
        Returns:
            List of validation warnings
        """
        warnings = []
        
        try:
            # Check if document has any content
            if not doc.paragraphs and not doc.tables:
                warnings.append("Document appears to be empty (no paragraphs or tables)")
            
            # Check for extremely large documents
            paragraph_count = len(doc.paragraphs)
            if paragraph_count > 10000:
                warnings.append(f"Document has many paragraphs ({paragraph_count}), processing may be slow")
            
            # Check for tables
            table_count = len(doc.tables)
            if table_count > 100:
                warnings.append(f"Document has many tables ({table_count}), processing may be slow")
            
            # Check for embedded objects (basic check)
            try:
                if hasattr(doc, 'inline_shapes'):
                    shape_count = len(doc.inline_shapes)
                    if shape_count > 50:
                        warnings.append(f"Document has many embedded objects ({shape_count})")
            except:
                pass  # Ignore if inline_shapes not accessible
            
        except Exception as e:
            warnings.append(f"Could not fully validate document structure: {str(e)}")
        
        return warnings
    
    def get_document_info(self, file_path: str) -> dict:
        """
        Get basic information about the DOCX document.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Dictionary with document information
        """
        try:
            doc = Document(file_path)
            
            info = {
                'paragraph_count': len(doc.paragraphs),
                'table_count': len(doc.tables),
                'has_core_properties': hasattr(doc, 'core_properties'),
                'file_size': Path(file_path).stat().st_size
            }
            
            # Try to get more detailed info
            try:
                if hasattr(doc, 'sections'):
                    info['section_count'] = len(doc.sections)
                
                if hasattr(doc, 'inline_shapes'):
                    info['embedded_object_count'] = len(doc.inline_shapes)
                    
            except:
                pass  # Ignore if not accessible
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting document info for {file_path}: {e}")
            return {'error': str(e)}