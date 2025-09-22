"""
Unified embedding generator for cross-modal content.

Implements a unified vector space mapping for text, image, and audio content
using sentence-transformers for text, CLIP for images, and text embeddings
for transcribed audio. Includes caching to avoid recomputation.
"""

import hashlib
import logging
from typing import List, Dict, Any, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import torch
    import clip
    from PIL import Image
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

from .base import MultimodalEmbeddingGenerator, EmbeddingCache, ModelLoadError, EmbeddingError
from .device_manager import DeviceManager
from .model_loader import ModelLoader
from .embedding_processors import EmbeddingProcessor, ImageProcessor
from .batch_processor import BatchProcessor
from ..models import ContentType
from ..config import EmbeddingConfig

logger = logging.getLogger(__name__)


class UnifiedEmbeddingGenerator(MultimodalEmbeddingGenerator):
    """
    Unified embedding generator that handles all content types.
    
    Maps text, image, and audio content into a unified vector space for
    cross-modal retrieval. Uses sentence-transformers for text, CLIP for images,
    and text embeddings for transcribed audio content.
    
    Features:
    - Cross-modal embedding generation
    - Unified vector space mapping
    - Embedding caching for performance
    - Batch processing support
    - Device optimization (CPU/GPU)
    """
    
    def __init__(self, config: EmbeddingConfig):
        """
        Initialize unified embedding generator.
        
        Args:
            config: Embedding configuration
        """
        super().__init__(config)
        
        if not MODELS_AVAILABLE:
            raise ImportError(
                "Required dependencies missing. Install with: "
                "pip install sentence-transformers torch clip-by-openai pillow"
            )
        
        # Device configuration
        self.device = DeviceManager.determine_device(config.device)
        
        # Initialize components
        self.model_loader = ModelLoader(config, self.device)
        self.embedding_processor = EmbeddingProcessor(config.embedding_dimension, self.device)
        
        # Initialize models (will be loaded on demand)
        self.text_model: Optional[SentenceTransformer] = None
        self.clip_model: Optional[Any] = None
        self.clip_preprocess: Optional[Any] = None
        self.image_processor: Optional[ImageProcessor] = None
        self.batch_processor: Optional[BatchProcessor] = None
        
        # Initialize cache
        self.cache = EmbeddingCache(
            cache_dir=config.embedding_cache_dir,
            enabled=config.cache_embeddings
        )
        
        # Performance tracking
        self._embedding_stats = {
            'total_embeddings': 0,
            'cache_hits': 0,
            'text_embeddings': 0,
            'image_embeddings': 0,
            'audio_embeddings': 0,
            'batch_operations': 0
        }
        
        # Model loading flags
        self._text_model_loaded = False
        self._clip_model_loaded = False
        
        logger.info(f"UnifiedEmbeddingGenerator initialized with device: {self.device}")
    
    def load_model(self) -> None:
        """
        Load all embedding models.
        
        Raises:
            ModelLoadError: If any model loading fails
        """
        try:
            # Load text model
            if not self._text_model_loaded:
                self.text_model = self.model_loader.load_text_model()
                self._text_model_loaded = True
            
            # Load CLIP model
            if not self._clip_model_loaded:
                self.clip_model, self.clip_preprocess = self.model_loader.load_clip_model()
                self.image_processor = ImageProcessor(self.clip_model, self.clip_preprocess, self.device)
                self._clip_model_loaded = True
            
            # Initialize batch processor
            if not self.batch_processor:
                self.batch_processor = BatchProcessor(
                    self.text_model, 
                    self.embedding_processor, 
                    self.device
                )
            
            self.is_loaded = True
            logger.info("All embedding models loaded successfully")
            
        except Exception as e:
            error_msg = f"Failed to load unified embedding models: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg, cause=e)
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text content into embedding vector.
        
        Args:
            text: Text content to encode
            
        Returns:
            Text embedding vector
            
        Raises:
            EmbeddingError: If text encoding fails
        """
        if not self._text_model_loaded:
            self.load_model()
        
        try:
            # Check cache first
            cache_key = self._compute_content_hash(text, ContentType.TEXT)
            cached_embedding = self.cache.get_embedding(cache_key)
            
            if cached_embedding is not None:
                self._embedding_stats['cache_hits'] += 1
                return cached_embedding
            
            # Generate embedding
            if not text or not text.strip():
                embedding = np.zeros(self.embedding_dimension, dtype=np.float32)
            else:
                embedding = self.text_model.encode(text, convert_to_numpy=True)
            
            # Validate and cache
            self.embedding_processor.validate_embedding(embedding)
            self.cache.store_embedding(cache_key, embedding)
            
            # Update stats
            self._embedding_stats['text_embeddings'] += 1
            self._embedding_stats['total_embeddings'] += 1
            
            return embedding
            
        except Exception as e:
            error_msg = f"Failed to encode text: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg, ContentType.TEXT, cause=e)
    
    def encode_image(self, image_path: str) -> np.ndarray:
        """
        Encode image content into embedding vector.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image embedding vector mapped to text space
            
        Raises:
            EmbeddingError: If image encoding fails
        """
        if not self._clip_model_loaded:
            self.load_model()
        
        try:
            # Check cache first
            cache_key = self._compute_content_hash(image_path, ContentType.IMAGE)
            cached_embedding = self.cache.get_embedding(cache_key)
            
            if cached_embedding is not None:
                self._embedding_stats['cache_hits'] += 1
                return cached_embedding
            
            # Process image
            clip_embedding = self.image_processor.process_image(image_path)
            
            # Map to text space
            embedding = self.embedding_processor.map_clip_to_text_space(clip_embedding)
            
            # Validate and cache
            self.embedding_processor.validate_embedding(embedding)
            self.cache.store_embedding(cache_key, embedding)
            
            # Update stats
            self._embedding_stats['image_embeddings'] += 1
            self._embedding_stats['total_embeddings'] += 1
            
            return embedding
            
        except Exception as e:
            error_msg = f"Failed to encode image {image_path}: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg, ContentType.IMAGE, cause=e)
    
    def encode_audio(self, audio_content: str) -> np.ndarray:
        """
        Encode audio content (transcription) into embedding vector.
        
        Args:
            audio_content: Transcribed audio text
            
        Returns:
            Audio embedding vector in unified space
            
        Raises:
            EmbeddingError: If audio encoding fails
        """
        if not self._text_model_loaded:
            self.load_model()
        
        try:
            # Check cache first
            cache_key = self._compute_content_hash(audio_content, ContentType.AUDIO)
            cached_embedding = self.cache.get_embedding(cache_key)
            
            if cached_embedding is not None:
                self._embedding_stats['cache_hits'] += 1
                return cached_embedding
            
            # Preprocess audio text
            processed_text = self.embedding_processor.preprocess_audio_text(audio_content)
            
            # Generate text embedding
            if not processed_text or not processed_text.strip():
                text_embedding = np.zeros(self.embedding_dimension, dtype=np.float32)
            else:
                text_embedding = self.text_model.encode(processed_text, convert_to_numpy=True)
            
            # Apply audio transformation
            embedding = self.embedding_processor.apply_audio_transformation(text_embedding)
            
            # Validate and cache
            self.embedding_processor.validate_embedding(embedding)
            self.cache.store_embedding(cache_key, embedding)
            
            # Update stats
            self._embedding_stats['audio_embeddings'] += 1
            self._embedding_stats['total_embeddings'] += 1
            
            return embedding
            
        except Exception as e:
            error_msg = f"Failed to encode audio content: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg, ContentType.AUDIO, cause=e)
    
    def generate_embedding(self, content: str, content_type: ContentType) -> np.ndarray:
        """
        Generate embedding for content based on its type.
        
        Args:
            content: Content to encode
            content_type: Type of content
            
        Returns:
            Embedding vector
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if content_type == ContentType.TEXT:
            return self.encode_text(content)
        elif content_type == ContentType.IMAGE:
            return self.encode_image(content)
        elif content_type == ContentType.AUDIO:
            return self.encode_audio(content)
        else:
            raise EmbeddingError(f"Unsupported content type: {content_type}")
    
    def generate_batch_embeddings(
        self, 
        contents: List[str], 
        content_types: List[ContentType]
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple content items efficiently.
        
        Args:
            contents: List of content strings
            content_types: List of corresponding content types
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingError: If batch generation fails
        """
        if not self.batch_processor:
            self.load_model()
        
        try:
            self._embedding_stats['batch_operations'] += 1
            embeddings = self.batch_processor.process_mixed_batch(contents, content_types)
            self._embedding_stats['total_embeddings'] += len(embeddings)
            return embeddings
            
        except Exception as e:
            error_msg = f"Batch embedding generation failed: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg, cause=e)
    
    def _compute_content_hash(self, content: str, content_type: ContentType) -> str:
        """
        Compute hash for content caching.
        
        Args:
            content: Content string
            content_type: Type of content
            
        Returns:
            Hash string for caching
        """
        # Include content type and model info in hash for uniqueness
        cache_key = f"{content_type.value}:{content}:{self.config.text_model_name}"
        
        if content_type == ContentType.IMAGE:
            cache_key += f":{self.config.clip_model_name}"
        
        return hashlib.sha256(cache_key.encode()).hexdigest()
    
    def get_supported_content_types(self) -> List[ContentType]:
        """Get list of supported content types."""
        return [ContentType.TEXT, ContentType.IMAGE, ContentType.AUDIO]
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the embedding generator.
        
        Returns:
            Dictionary with performance metrics
        """
        cache_stats = self.cache.get_cache_stats() if self.cache else {}
        
        return {
            **self._embedding_stats,
            **cache_stats,
            'device': self.device,
            'models_loaded': {
                'text': self._text_model_loaded,
                'clip': self._clip_model_loaded
            },
            'embedding_dimension': self.embedding_dimension,
            'cache_enabled': self.cache.enabled if self.cache else False
        }
    
    def clear_cache(self) -> None:
        """Clear embedding cache."""
        if self.cache:
            self.cache.clear_cache()
        logger.info("Unified embedding cache cleared")
    
    def __del__(self):
        """Cleanup resources when generator is destroyed."""
        try:
            if hasattr(self, 'cache') and self.cache:
                self.cache.close()
        except Exception:
            pass  # Ignore cleanup errors