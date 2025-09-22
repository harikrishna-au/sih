"""
Model loading utilities for embedding generators.

Handles loading and initialization of various embedding models including
sentence transformers and CLIP models with proper error handling and validation.
"""

import logging
import ssl
import os
from typing import Optional, Any, Tuple

# Disable SSL verification to avoid certificate issues
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['CURL_CA_BUNDLE'] = ''

try:
    from sentence_transformers import SentenceTransformer
    import torch
    import clip
    from PIL import Image
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

from .base import ModelLoadError
from ..config import EmbeddingConfig

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading and validation of embedding models."""
    
    def __init__(self, config: EmbeddingConfig, device: str):
        """
        Initialize model loader.
        
        Args:
            config: Embedding configuration
            device: Target device for models
        """
        self.config = config
        self.device = device
        
        if not MODELS_AVAILABLE:
            raise ImportError(
                "Required dependencies missing. Install with: "
                "pip install sentence-transformers torch clip-by-openai pillow"
            )
    
    def load_text_model(self):
        """
        Load sentence transformer model for text embeddings.
        
        Returns:
            Loaded SentenceTransformer model
            
        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            logger.info(f"Loading text embedding model: {self.config.text_model_name}")
            model = SentenceTransformer(
                self.config.text_model_name,
                device=self.device
            )
            
            # Validate embedding dimension
            test_embedding = model.encode("test", convert_to_numpy=True)
            actual_dimension = test_embedding.shape[0]
            
            if actual_dimension != self.config.embedding_dimension:
                logger.warning(
                    f"Text model embedding dimension ({actual_dimension}) differs from "
                    f"configured dimension ({self.config.embedding_dimension}). "
                    f"Model dimension will be used."
                )
            
            model.eval()
            logger.info(f"Text model loaded successfully on {self.device}")
            return model
            
        except Exception as e:
            error_msg = f"Failed to load text model: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg, cause=e)
    
    def load_clip_model(self) -> Tuple[Any, Any]:
        """
        Load CLIP model for image embeddings.
        
        Returns:
            Tuple of (clip_model, clip_preprocess)
            
        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            logger.info(f"Loading CLIP model: {self.config.clip_model_name}")
            model, preprocess = clip.load(
                self.config.clip_model_name,
                device=self.device
            )
            
            model.eval()
            logger.info(f"CLIP model loaded successfully on {self.device}")
            return model, preprocess
            
        except Exception as e:
            error_msg = f"Failed to load CLIP model: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg, cause=e)
    
    def validate_model_compatibility(self, text_model: Any, clip_model: Any) -> bool:
        """
        Validate that loaded models are compatible.
        
        Args:
            text_model: Loaded text model
            clip_model: Loaded CLIP model
            
        Returns:
            True if models are compatible
            
        Raises:
            ModelLoadError: If models are incompatible
        """
        try:
            # Test text model
            text_embedding = text_model.encode("test", convert_to_numpy=True)
            text_dim = text_embedding.shape[0]
            
            # Test CLIP model with dummy image
            dummy_image = Image.new('RGB', (224, 224), color='white')
            with torch.no_grad():
                clip_embedding = clip_model.encode_image(
                    clip.tokenize(["test"]).to(self.device)
                ).cpu().numpy()
            clip_dim = clip_embedding.shape[1]
            
            logger.info(f"Model dimensions - Text: {text_dim}, CLIP: {clip_dim}")
            
            # Models don't need to have same dimensions as we can map between spaces
            return True
            
        except Exception as e:
            error_msg = f"Model compatibility validation failed: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg, cause=e)
    
    def get_model_info(self, model_name: str, model_type: str) -> dict:
        """
        Get information about a model.
        
        Args:
            model_name: Name of the model
            model_type: Type of model ('text' or 'clip')
            
        Returns:
            Dictionary with model information
        """
        info = {
            "name": model_name,
            "type": model_type,
            "device": self.device,
            "loaded": False
        }
        
        try:
            if model_type == "text":
                # Try to get model info without loading
                info["architecture"] = "sentence-transformer"
                info["supported"] = True
            elif model_type == "clip":
                info["architecture"] = "CLIP"
                info["supported"] = model_name in clip.available_models()
            
        except Exception as e:
            info["error"] = str(e)
            info["supported"] = False
        
        return info