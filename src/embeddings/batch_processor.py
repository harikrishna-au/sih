"""
Batch processing utilities for efficient embedding generation.

Provides optimized batch processing for multiple content items
with proper error handling and performance optimization.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import torch
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

from ..models import ContentType
from .base import EmbeddingError
from .embedding_processors import EmbeddingProcessor

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Handles batch processing of embeddings for improved performance."""
    
    def __init__(
        self, 
        text_model: Optional[Any],
        embedding_processor: EmbeddingProcessor,
        device: str,
        batch_size: int = 32
    ):
        """
        Initialize batch processor.
        
        Args:
            text_model: Loaded text model
            embedding_processor: Embedding processor instance
            device: Device for computations
            batch_size: Batch size for processing
        """
        self.text_model = text_model
        self.embedding_processor = embedding_processor
        self.device = device
        self.batch_size = batch_size
    
    def process_text_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Process multiple texts efficiently in batches.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingError: If batch processing fails
        """
        if not self.text_model:
            raise EmbeddingError("Text model not loaded", ContentType.TEXT)
        
        try:
            embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                # Filter out empty texts
                valid_texts = [text for text in batch_texts if text and text.strip()]
                
                if not valid_texts:
                    # Add zero embeddings for empty texts
                    batch_embeddings = [
                        np.zeros(self.embedding_processor.embedding_dimension, dtype=np.float32)
                        for _ in batch_texts
                    ]
                else:
                    # Generate embeddings for valid texts
                    batch_embeddings = self.text_model.encode(
                        valid_texts,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        batch_size=len(valid_texts)
                    )
                    
                    # Handle case where some texts in batch were empty
                    if len(valid_texts) != len(batch_texts):
                        full_batch_embeddings = []
                        valid_idx = 0
                        
                        for text in batch_texts:
                            if text and text.strip():
                                full_batch_embeddings.append(batch_embeddings[valid_idx])
                                valid_idx += 1
                            else:
                                full_batch_embeddings.append(
                                    np.zeros(self.embedding_processor.embedding_dimension, dtype=np.float32)
                                )
                        
                        batch_embeddings = full_batch_embeddings
                
                embeddings.extend(batch_embeddings)
            
            return embeddings
            
        except Exception as e:
            error_msg = f"Batch text encoding failed: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg, ContentType.TEXT, cause=e)
    
    def process_audio_batch(self, audio_contents: List[str]) -> List[np.ndarray]:
        """
        Process multiple audio transcriptions efficiently.
        
        Args:
            audio_contents: List of transcribed audio texts
            
        Returns:
            List of audio embedding vectors
            
        Raises:
            EmbeddingError: If batch processing fails
        """
        try:
            # Preprocess all audio texts
            processed_texts = [
                self.embedding_processor.preprocess_audio_text(content)
                for content in audio_contents
            ]
            
            # Generate text embeddings in batch
            text_embeddings = self.process_text_batch(processed_texts)
            
            # Apply audio transformations
            audio_embeddings = []
            for text_embedding in text_embeddings:
                audio_embedding = self.embedding_processor.apply_audio_transformation(text_embedding)
                audio_embeddings.append(audio_embedding)
            
            return audio_embeddings
            
        except Exception as e:
            error_msg = f"Batch audio encoding failed: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg, ContentType.AUDIO, cause=e)
    
    def process_mixed_batch(
        self, 
        contents: List[str], 
        content_types: List[ContentType]
    ) -> List[np.ndarray]:
        """
        Process mixed content types in an optimized way.
        
        Args:
            contents: List of content strings
            content_types: List of corresponding content types
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingError: If batch processing fails
        """
        if len(contents) != len(content_types):
            raise ValueError("Contents and content_types must have same length")
        
        try:
            embeddings = [None] * len(contents)
            
            # Group by content type for batch processing
            type_groups = {}
            for i, (content, content_type) in enumerate(zip(contents, content_types)):
                if content_type not in type_groups:
                    type_groups[content_type] = []
                type_groups[content_type].append((i, content))
            
            # Process each content type in batches
            for content_type, items in type_groups.items():
                indices, batch_contents = zip(*items)
                
                if content_type == ContentType.TEXT:
                    batch_embeddings = self.process_text_batch(list(batch_contents))
                elif content_type == ContentType.AUDIO:
                    batch_embeddings = self.process_audio_batch(list(batch_contents))
                else:
                    # For unsupported types, process individually
                    batch_embeddings = []
                    for content in batch_contents:
                        # This would need to be handled by the main generator
                        batch_embeddings.append(
                            np.zeros(self.embedding_processor.embedding_dimension, dtype=np.float32)
                        )
                
                # Place embeddings back in original order
                for idx, embedding in zip(indices, batch_embeddings):
                    embeddings[idx] = embedding
            
            return embeddings
            
        except Exception as e:
            error_msg = f"Mixed batch processing failed: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg, cause=e)
    
    def optimize_batch_size(self, content_count: int, content_type: ContentType) -> int:
        """
        Optimize batch size based on content count and type.
        
        Args:
            content_count: Number of items to process
            content_type: Type of content
            
        Returns:
            Optimized batch size
        """
        base_batch_size = self.batch_size
        
        # Adjust based on content type
        if content_type == ContentType.IMAGE:
            # Images are more memory intensive
            base_batch_size = max(1, base_batch_size // 4)
        elif content_type == ContentType.AUDIO:
            # Audio processing involves text preprocessing
            base_batch_size = max(1, base_batch_size // 2)
        
        # Adjust based on device
        if self.device == "cpu":
            base_batch_size = max(1, base_batch_size // 2)
        
        # Don't exceed content count
        return min(base_batch_size, content_count)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get batch processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        return {
            "batch_size": self.batch_size,
            "device": self.device,
            "text_model_loaded": self.text_model is not None,
            "embedding_dimension": self.embedding_processor.embedding_dimension
        }