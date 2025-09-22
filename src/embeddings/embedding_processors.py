"""
Embedding processing utilities for different content types.

Provides specialized processing for text, image, and audio embeddings
including transformations, mappings, and preprocessing operations.
"""

import logging
import re
import numpy as np
from typing import List, Optional, Any
from PIL import Image

try:
    import torch
    import clip
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..models import ContentType
from .base import EmbeddingError

logger = logging.getLogger(__name__)


class EmbeddingProcessor:
    """Handles processing and transformation of embeddings."""
    
    def __init__(self, embedding_dimension: int, device: str):
        """
        Initialize embedding processor.
        
        Args:
            embedding_dimension: Target embedding dimension
            device: Device for computations
        """
        self.embedding_dimension = embedding_dimension
        self.device = device
    
    def map_clip_to_text_space(self, clip_embedding: np.ndarray) -> np.ndarray:
        """
        Map CLIP embedding to text embedding space.
        
        Args:
            clip_embedding: CLIP embedding vector
            
        Returns:
            Mapped embedding in text space
        """
        try:
            # Simple linear mapping (in practice, this could be learned)
            # For now, we pad or truncate to match text embedding dimension
            clip_dim = clip_embedding.shape[0]
            
            if clip_dim == self.embedding_dimension:
                return clip_embedding
            elif clip_dim > self.embedding_dimension:
                # Truncate
                return clip_embedding[:self.embedding_dimension]
            else:
                # Pad with zeros
                padded = np.zeros(self.embedding_dimension, dtype=clip_embedding.dtype)
                padded[:clip_dim] = clip_embedding
                return padded
                
        except Exception as e:
            logger.error(f"Error mapping CLIP embedding: {e}")
            raise EmbeddingError(f"Failed to map CLIP embedding: {e}", ContentType.IMAGE)
    
    def preprocess_audio_text(self, audio_text: str) -> str:
        """
        Preprocess transcribed audio text for embedding.
        
        Args:
            audio_text: Transcribed audio text
            
        Returns:
            Preprocessed text
        """
        if not audio_text or not audio_text.strip():
            return ""
        
        # Clean up common transcription artifacts
        processed = audio_text.strip()
        
        # Remove excessive whitespace
        processed = re.sub(r'\s+', ' ', processed)
        
        # Remove common filler words that might be transcription artifacts
        filler_words = ['um', 'uh', 'er', 'ah', 'like', 'you know']
        for filler in filler_words:
            processed = re.sub(rf'\b{filler}\b', '', processed, flags=re.IGNORECASE)
        
        # Clean up punctuation artifacts
        processed = re.sub(r'[.,]{2,}', '.', processed)
        processed = re.sub(r'\s+', ' ', processed)
        
        # Add audio context marker
        processed = f"[AUDIO] {processed.strip()}"
        
        return processed
    
    def apply_audio_transformation(self, text_embedding: np.ndarray) -> np.ndarray:
        """
        Apply transformation to distinguish audio embeddings from text.
        
        Args:
            text_embedding: Text embedding of transcribed audio
            
        Returns:
            Transformed embedding for audio content
        """
        try:
            # Apply a simple transformation to create audio-specific embedding space
            # In practice, this could be a learned transformation
            
            # Method 1: Apply rotation matrix (simple orthogonal transformation)
            transformation_matrix = self._get_audio_transformation_matrix()
            transformed = np.dot(text_embedding, transformation_matrix)
            
            # Method 2: Add audio-specific bias
            audio_bias = self._get_audio_bias_vector()
            transformed = transformed + audio_bias
            
            # Normalize to maintain embedding properties
            norm = np.linalg.norm(transformed)
            if norm > 0:
                transformed = transformed / norm
            
            return transformed
            
        except Exception as e:
            logger.error(f"Error applying audio transformation: {e}")
            raise EmbeddingError(f"Failed to transform audio embedding: {e}", ContentType.AUDIO)
    
    def _get_audio_transformation_matrix(self) -> np.ndarray:
        """Get transformation matrix for audio embeddings."""
        # Create a simple rotation matrix
        # In practice, this could be learned from data
        np.random.seed(42)  # Fixed seed for consistency
        matrix = np.random.orthogonal(self.embedding_dimension)
        return matrix.astype(np.float32)
    
    def _get_audio_bias_vector(self) -> np.ndarray:
        """Get bias vector for audio embeddings."""
        # Create a small bias vector to shift audio embeddings
        np.random.seed(123)  # Fixed seed for consistency
        bias = np.random.normal(0, 0.1, self.embedding_dimension)
        return bias.astype(np.float32)
    
    def validate_embedding(self, embedding: np.ndarray) -> bool:
        """
        Validate that embedding is well-formed.
        
        Args:
            embedding: Embedding vector to validate
            
        Returns:
            True if embedding is valid
            
        Raises:
            EmbeddingError: If embedding is invalid
        """
        try:
            # Check if embedding is numpy array
            if not isinstance(embedding, np.ndarray):
                raise ValueError("Embedding must be numpy array")
            
            # Check dimension
            if embedding.shape[0] != self.embedding_dimension:
                raise ValueError(
                    f"Embedding dimension {embedding.shape[0]} does not match "
                    f"expected dimension {self.embedding_dimension}"
                )
            
            # Check for NaN or infinite values
            if np.isnan(embedding).any():
                raise ValueError("Embedding contains NaN values")
            
            if np.isinf(embedding).any():
                raise ValueError("Embedding contains infinite values")
            
            # Check if embedding is all zeros (might indicate an error)
            if np.allclose(embedding, 0):
                logger.warning("Embedding is all zeros")
            
            return True
            
        except Exception as e:
            logger.error(f"Embedding validation failed: {e}")
            raise EmbeddingError(f"Invalid embedding: {e}")
    
    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalize embedding vector.
        
        Args:
            embedding: Embedding vector to normalize
            
        Returns:
            Normalized embedding
        """
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        else:
            logger.warning("Cannot normalize zero vector")
            return embedding
    
    def compute_embedding_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            
            # Clamp to [-1, 1] range due to floating point precision
            similarity = np.clip(similarity, -1.0, 1.0)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0


class ImageProcessor:
    """Specialized processor for image embeddings."""
    
    def __init__(self, clip_model: Any, clip_preprocess: Any, device: str):
        """
        Initialize image processor.
        
        Args:
            clip_model: Loaded CLIP model
            clip_preprocess: CLIP preprocessing function
            device: Device for computations
        """
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.device = device
    
    def process_image(self, image_path: str) -> np.ndarray:
        """
        Process image and extract CLIP embedding.
        
        Args:
            image_path: Path to image file
            
        Returns:
            CLIP embedding vector
            
        Raises:
            EmbeddingError: If image processing fails
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_tensor)
                embedding = image_features.cpu().numpy().flatten()
            
            return embedding
            
        except Exception as e:
            error_msg = f"Failed to process image {image_path}: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg, ContentType.IMAGE, cause=e)
    
    def validate_image(self, image_path: str) -> bool:
        """
        Validate that image can be processed.
        
        Args:
            image_path: Path to image file
            
        Returns:
            True if image is valid
        """
        try:
            image = Image.open(image_path)
            # Check if image can be converted to RGB
            image.convert('RGB')
            return True
        except Exception as e:
            logger.warning(f"Invalid image {image_path}: {e}")
            return False