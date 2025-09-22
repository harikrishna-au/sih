"""Embedding generation components."""

from .base import EmbeddingGenerator, MultimodalEmbeddingGenerator, EmbeddingCache, ModelLoadError, EmbeddingError

# Import embedding generators conditionally to avoid dependency issues
__all__ = [
    'EmbeddingGenerator',
    'MultimodalEmbeddingGenerator', 
    'EmbeddingCache',
    'ModelLoadError',
    'EmbeddingError'
]

try:
    from .text_embedding_generator import SentenceTransformerEmbeddingGenerator
    __all__.append('SentenceTransformerEmbeddingGenerator')
except ImportError:
    pass

try:
    from .unified_embedding_generator import UnifiedEmbeddingGenerator
    __all__.append('UnifiedEmbeddingGenerator')
except ImportError:
    pass

try:
    from .device_manager import DeviceManager
    __all__.append('DeviceManager')
except ImportError:
    pass

try:
    from .embedding_processors import EmbeddingProcessor, ImageProcessor
    __all__.extend(['EmbeddingProcessor', 'ImageProcessor'])
except ImportError:
    pass