"""
Document processors package.

Provides document processing capabilities for various file formats
including PDF, DOCX, images, and audio files.
"""

from .base import DocumentProcessor, ProcessingError, UnsupportedFormatError, FileValidationError

# Import processors conditionally to avoid dependency issues
__all__ = [
    'DocumentProcessor',
    'ProcessingError',
    'UnsupportedFormatError',
    'FileValidationError'
]

try:
    from .router import DocumentRouter
    __all__.append('DocumentRouter')
except ImportError:
    pass

try:
    from .pdf_processor import PDFProcessor
    __all__.append('PDFProcessor')
except ImportError:
    pass

try:
    from .docx_processor import DOCXProcessor
    __all__.append('DOCXProcessor')
except ImportError:
    pass

try:
    from .image_processor import ImageProcessor
    __all__.append('ImageProcessor')
except ImportError:
    pass

try:
    from .ocr_engine import OCREngine, OCRResult
    __all__.extend(['OCREngine', 'OCRResult'])
except ImportError:
    pass

try:
    from .audio_processor import AudioProcessor
    __all__.append('AudioProcessor')
except ImportError:
    pass