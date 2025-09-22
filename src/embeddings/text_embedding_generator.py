"""
Text embedding generator using sentence-transformers library.

Implements efficient text embedding generation with batch processing,
normalization, and dimension validation for the multimodal RAG system.
"""

import hashlib
import logging
from typing import List, Optional, Dict, Any
import numpy as np
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError as e:
    raise ImportError(
        "sentence-transformers and torch are required for text embedding generation. "
        "Install with: pip install sentence-transformers torch"
    ) from e

from .base import TextEmbeddingGenerator, EmbeddingCache, ModelLoadError, EmbeddingError
from .embedding_processors import EmbeddingProcessor
from ..models import ContentType
from ..config import EmbeddingConfig


logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddingGenerator(TextEmbeddingGenerator):
    """
    Text embedding generator using sentence-transformers library.
    
    Provides efficient text embedding generation with support for:
    - Batch processing for optimal performance
    - Embedding normalization and validation
    - Caching to avoid recomputation
    - Multiple device support (CPU/GPU)
    """
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.model: Optional[SentenceTransformer] = None
        self.device = self._determine_device()
        self.cache = EmbeddingCache(
            cache_dir=config.embedding_cache_dir,
            enabled=config.cache_embeddings
        )
        
        # Initialize embedding processor for validation
        self.embedding_processor = EmbeddingProcessor(config.embedding_dimension, self.device)
        
        # Performance tracking
        self._embedding_stats = {
            'total_embeddings': 0,
            'cache_hits': 0,
            'batch_operations': 0
        }
    
    def _determine_device(self) -> str:
        """
        Determine the best available device for embedding generation.
        
        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        if self.config.device != "auto":
            return self.config.device
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_model(self) -> None:
        """
        Load the sentence transformer model.
        
        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            logger.info(f"Loading text embedding model: {self.config.text_model_name}")
            logger.info(f"Using device: {self.device}")
            
            self.model = SentenceTransformer(
                self.config.text_model_name,
                device=self.device
            )
            
            # Set model to evaluation mode for consistency
            self.model.eval()
            
            self.is_loaded = True
            logger.info(
                f"Successfully loaded text embedding model. "
                f"Dimension: {self.embedding_dimension}, Device: {self.device}"
            )
            
        except Exception as e:
            error_msg = f"Failed to load text embedding model '{self.config.text_model_name}': {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg, cause=e)
    
    def _validate_and_update_dimension(self, embedding: np.ndarray) -> None:
        """
        Validate and potentially update embedding dimension based on actual model output.
        
        Args:
            embedding: Sample embedding from the model
        """
        actual_dimension = embedding.shape[0]
        
        if actual_dimension != self.config.embedding_dimension:
            logger.warning(
                f"Model embedding dimension ({actual_dimension}) differs from "
                f"configured dimension ({self.config.embedding_dimension}). "
                f"Updating configuration."
            )
            self.embedding_dimension = actual_dimension
            self.config.embedding_dimension = actual_dimension
            
            # Update embedding processor dimension too
            if hasattr(self, 'embedding_processor'):
                self.embedding_processor = EmbeddingProcessor(actual_dimension, self.device)
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode single text into embedding vector.
        
        Args:
            text: Text content to encode
            
        Returns:
            Normalized embedding vector
            
        Raises:
            EmbeddingError: If encoding fails
        """
        if not self.is_loaded:
            self.load_model()
        
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return np.zeros(self.embedding_dimension, dtype=np.float32)
        
        # Check cache first
        content_hash = self._compute_content_hash(text)
        cached_embedding = self.cache.get_embedding(content_hash)
        if cached_embedding is not None:
            self._embedding_stats['cache_hits'] += 1
            return cached_embedding
        
        try:
            # Truncate text if too long
            if len(text) > self.config.max_sequence_length * 4:  # Rough character estimate
                text = text[:self.config.max_sequence_length * 4]
                logger.debug(f"Truncated long text to {len(text)} characters")
            
            # Generate embedding
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=False,  # We handle normalization ourselves
                show_progress_bar=False
            )
            
            # On first use, validate and potentially update dimension
            if self._embedding_stats['total_embeddings'] == 0:
                self._validate_and_update_dimension(embedding)
            
            # Validate embedding
            if not self._validate_embedding(embedding):
                logger.warning("Generated invalid embedding, returning zero vector")
                embedding = np.zeros(self.embedding_dimension, dtype=np.float32)
            
            # Normalize if configured
            embedding = self.normalize_embedding(embedding)
            
            # Cache the embedding
            self.cache.store_embedding(content_hash, embedding)
            
            self._embedding_stats['total_embeddings'] += 1
            return embedding
            
        except Exception as e:
            error_msg = f"Failed to encode text: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg, ContentType.TEXT, cause=e)
    
    def encode_text_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Encode multiple texts into embedding vectors efficiently.
        
        Args:
            texts: List of text content to encode
            
        Returns:
            List of normalized embedding vectors
            
        Raises:
            EmbeddingError: If batch encoding fails
        """
        if not self.is_loaded:
            self.load_model()
        
        if not texts:
            return []
        
        # Filter out empty texts and track indices
        valid_texts = []
        valid_indices = []
        text_hashes = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                # Check cache
                content_hash = self._compute_content_hash(text)
                cached_embedding = self.cache.get_embedding(content_hash)
                
                if cached_embedding is None:
                    valid_texts.append(text)
                    valid_indices.append(i)
                    text_hashes.append(content_hash)
                else:
                    self._embedding_stats['cache_hits'] += 1
        
        # Initialize result array
        embeddings = [np.zeros(self.embedding_dimension, dtype=np.float32) for _ in texts]
        
        # Fill cached embeddings
        for i, text in enumerate(texts):
            if text and text.strip():
                content_hash = self._compute_content_hash(text)
                cached_embedding = self.cache.get_embedding(content_hash)
                if cached_embedding is not None:
                    embeddings[i] = cached_embedding
        
        # Process non-cached texts in batches
        if valid_texts:
            try:
                # Truncate long texts
                processed_texts = []
                for text in valid_texts:
                    if len(text) > self.config.max_sequence_length * 4:
                        text = text[:self.config.max_sequence_length * 4]
                    processed_texts.append(text)
                
                # Generate embeddings in batches
                batch_size = self.config.batch_size
                for batch_start in range(0, len(processed_texts), batch_size):
                    batch_end = min(batch_start + batch_size, len(processed_texts))
                    batch_texts = processed_texts[batch_start:batch_end]
                    batch_indices = valid_indices[batch_start:batch_end]
                    batch_hashes = text_hashes[batch_start:batch_end]
                    
                    # Generate batch embeddings
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        normalize_embeddings=False,
                        show_progress_bar=False,
                        batch_size=len(batch_texts)
                    )
                    
                    # On first use, validate and potentially update dimension
                    if self._embedding_stats['total_embeddings'] == 0 and len(batch_embeddings) > 0:
                        self._validate_and_update_dimension(batch_embeddings[0])
                    
                    # Process each embedding in the batch
                    for j, (embedding, original_idx, content_hash) in enumerate(
                        zip(batch_embeddings, batch_indices, batch_hashes)
                    ):
                        # Validate embedding
                        if not self._validate_embedding(embedding):
                            logger.warning(f"Invalid embedding generated for text at index {original_idx}")
                            embedding = np.zeros(self.embedding_dimension, dtype=np.float32)
                        else:
                            # Normalize if configured
                            embedding = self.normalize_embedding(embedding)
                            
                            # Cache the embedding
                            self.cache.store_embedding(content_hash, embedding)
                        
                        embeddings[original_idx] = embedding
                
                self._embedding_stats['total_embeddings'] += len(valid_texts)
                self._embedding_stats['batch_operations'] += 1
                
            except Exception as e:
                error_msg = f"Failed to encode text batch: {str(e)}"
                logger.error(error_msg)
                raise EmbeddingError(error_msg, ContentType.TEXT, cause=e)
        
        return embeddings
    
    def _compute_content_hash(self, content: str) -> str:
        """
        Compute hash for content caching.
        
        Args:
            content: Content to hash
            
        Returns:
            Content hash string
        """
        # Include model name and config in hash to avoid cache conflicts
        cache_key = f"{self.config.text_model_name}:{content}:{self.config.normalize_embeddings}"
        return hashlib.sha256(cache_key.encode()).hexdigest()
    
    def _validate_embedding(self, embedding: np.ndarray) -> bool:
        """
        Validate embedding for correctness.
        
        Args:
            embedding: Embedding vector to validate
            
        Returns:
            True if embedding is valid, False otherwise
        """
        if embedding is None:
            return False
        
        if not isinstance(embedding, np.ndarray):
            return False
        
        if embedding.shape[0] != self.embedding_dimension:
            return False
        
        if np.isnan(embedding).any():
            return False
        
        if np.isinf(embedding).any():
            return False
        
        if np.allclose(embedding, 0):
            return False
        
        return True

    def get_embedding_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the embedding generator.
        
        Returns:
            Dictionary containing performance metrics
        """
        cache_hit_rate = 0.0
        if self._embedding_stats['total_embeddings'] > 0:
            cache_hit_rate = (
                self._embedding_stats['cache_hits'] / 
                (self._embedding_stats['total_embeddings'] + self._embedding_stats['cache_hits'])
            )
        
        return {
            **self._embedding_stats,
            'cache_hit_rate': cache_hit_rate,
            'model_name': self.config.text_model_name,
            'device': self.device,
            'embedding_dimension': self.embedding_dimension,
            'is_loaded': self.is_loaded
        }
    
    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self.cache.clear_cache()
        logger.info("Embedding cache cleared")
    
    def __del__(self):
        """Cleanup resources when generator is destroyed."""
        if hasattr(self, 'model') and self.model is not None:
            # Clear GPU memory if using CUDA
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()