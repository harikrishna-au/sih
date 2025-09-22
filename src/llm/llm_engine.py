"""
Local LLM engine implementation using llama-cpp-python.

This module provides the LLMEngine class that loads and manages
quantized LLaMA 2 models for offline response generation with
context management and prompt engineering for RAG applications.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

try:
    from llama_cpp import Llama
except ImportError:
    raise ImportError(
        "llama-cpp-python is required for LLM functionality. "
        "Install it with: pip install llama-cpp-python"
    )

from ..config import LLMConfig
from ..models import RetrievalResult


logger = logging.getLogger(__name__)


class LLMEngine:
    """
    Local LLM engine that loads and manages quantized LLaMA 2 models.
    
    Provides context management, prompt engineering for RAG, and
    response generation with configurable parameters. Designed for
    offline operation with quantized models for efficient resource usage.
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialize the LLM engine with configuration.
        
        Args:
            config: LLM configuration containing model path and parameters
        """
        self.config = config
        self.model: Optional[Llama] = None
        self._model_loaded = False
        self._model_info: Dict[str, Any] = {}
        
        # Validate configuration
        self._validate_config()
        
        # Load model if path exists
        if os.path.exists(self.config.model_path):
            self.load_model()
        else:
            logger.warning(f"Model file not found at {self.config.model_path}")
    
    def _validate_config(self) -> None:
        """Validate LLM configuration parameters."""
        if not self.config.model_path:
            raise ValueError("model_path cannot be empty")
        
        if self.config.max_context_length <= 0:
            raise ValueError("max_context_length must be positive")
        
        if self.config.max_response_length <= 0:
            raise ValueError("max_response_length must be positive")
        
        if not (0 <= self.config.temperature <= 2):
            raise ValueError("temperature must be between 0 and 2")
        
        if not (0 <= self.config.top_p <= 1):
            raise ValueError("top_p must be between 0 and 1")
        
        if self.config.top_k <= 0:
            raise ValueError("top_k must be positive")
    
    def load_model(self) -> bool:
        """
        Load the quantized LLaMA model from the configured path.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading LLM model from {self.config.model_path}")
            start_time = time.time()
            
            # Determine number of threads
            n_threads = self.config.n_threads
            if n_threads == -1:
                n_threads = os.cpu_count() or 4
            
            # Load model with configuration
            self.model = Llama(
                model_path=self.config.model_path,
                n_ctx=self.config.max_context_length,
                n_threads=n_threads,
                verbose=False,
                use_mmap=True,
                use_mlock=False,
            )
            
            load_time = time.time() - start_time
            self._model_loaded = True
            
            # Store model information
            self._model_info = {
                "model_path": self.config.model_path,
                "load_time": load_time,
                "context_length": self.config.max_context_length,
                "threads": n_threads,
                "quantization": self.config.quantization,
            }
            
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self._model_loaded = False
            return False
    
    def is_model_loaded(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        return self._model_loaded and self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return self._model_info.copy()
    
    def generate_response(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a response using the loaded LLM.
        
        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate (uses config default if None)
            temperature: Sampling temperature (uses config default if None)
            top_p: Top-p sampling parameter (uses config default if None)
            top_k: Top-k sampling parameter (uses config default if None)
            stop_sequences: List of sequences that stop generation
            
        Returns:
            str: Generated response text
            
        Raises:
            RuntimeError: If model is not loaded
        """
        if not self.is_model_loaded():
            raise RuntimeError("Model is not loaded. Call load_model() first.")
        
        # Use config defaults if parameters not provided
        max_tokens = max_tokens or self.config.max_response_length
        temperature = temperature if temperature is not None else self.config.temperature
        top_p = top_p if top_p is not None else self.config.top_p
        top_k = top_k if top_k is not None else self.config.top_k
        
        # Default stop sequences
        if stop_sequences is None:
            stop_sequences = ["</s>", "[INST]", "[/INST]"]
        
        try:
            logger.debug(f"Generating response with max_tokens={max_tokens}, "
                        f"temperature={temperature}, top_p={top_p}, top_k={top_k}")
            
            start_time = time.time()
            
            # Generate response
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop_sequences,
                echo=False,
            )
            
            generation_time = time.time() - start_time
            
            # Extract text from response
            response_text = response["choices"][0]["text"].strip()
            
            logger.debug(f"Generated {len(response_text)} characters in {generation_time:.2f}s")
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error during response generation: {str(e)}")
            raise RuntimeError(f"Response generation failed: {str(e)}")
    
    def generate_with_context(
        self,
        context: List[str],
        query: str,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate a response using provided context and query.
        
        Args:
            context: List of context strings to include in prompt
            query: User query to answer
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            str: Generated response text
        """
        # Build context string
        context_str = "\n\n".join(context)
        
        # Create prompt using template
        prompt = self.config.citation_prompt_template.format(
            context=context_str,
            question=query
        )
        
        # Manage context length
        prompt = self._manage_context_length(prompt, max_tokens or self.config.max_response_length)
        
        return self.generate_response(prompt, max_tokens=max_tokens, **kwargs)
    
    def generate_with_citations(
        self,
        retrieval_results: List[RetrievalResult],
        query: str,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a response with citation information from retrieval results.
        
        Args:
            retrieval_results: List of retrieval results to use as context
            query: User query to answer
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Tuple[str, Dict[str, Any]]: Generated response and metadata
        """
        # Build context with citation markers
        context_parts = []
        citation_map = {}
        
        for i, result in enumerate(retrieval_results, 1):
            citation_id = f"[{i}]"
            context_parts.append(f"{citation_id} {result.content}")
            citation_map[citation_id] = {
                "chunk_id": result.chunk_id,
                "source_file": result.source_location.file_path,
                "similarity_score": result.similarity_score,
                "content_type": result.content_type.value,
            }
        
        # Generate response
        response = self.generate_with_context(
            context_parts, query, max_tokens=max_tokens, **kwargs
        )
        
        # Create metadata
        metadata = {
            "citation_map": citation_map,
            "context_length": len("\n\n".join(context_parts)),
            "num_sources": len(retrieval_results),
            "query": query,
        }
        
        return response, metadata
    
    def _manage_context_length(self, prompt: str, max_response_tokens: int) -> str:
        """
        Manage context length to fit within model limits.
        
        Args:
            prompt: Input prompt
            max_response_tokens: Maximum tokens reserved for response
            
        Returns:
            str: Truncated prompt if necessary
        """
        # Rough token estimation (1 token ≈ 4 characters)
        estimated_prompt_tokens = len(prompt) // 4
        available_tokens = self.config.max_context_length - max_response_tokens - 100  # Buffer
        
        if estimated_prompt_tokens <= available_tokens:
            return prompt
        
        logger.warning(f"Prompt too long ({estimated_prompt_tokens} tokens), truncating to fit")
        
        if self.config.context_window_strategy == "sliding":
            # Keep the beginning and end, truncate middle
            target_chars = available_tokens * 4
            if len(prompt) > target_chars:
                keep_start = target_chars // 3
                keep_end = target_chars // 3
                truncated = (
                    prompt[:keep_start] + 
                    "\n\n[... content truncated ...]\n\n" + 
                    prompt[-keep_end:]
                )
                return truncated
        else:
            # Simple truncation from the end
            target_chars = available_tokens * 4
            return prompt[:target_chars]
        
        return prompt
    
    def validate_response_quality(
        self,
        response: str,
        context: List[str],
        query: str
    ) -> float:
        """
        Validate the quality of a generated response.
        
        Args:
            response: Generated response text
            context: Context used for generation
            query: Original query
            
        Returns:
            float: Quality score between 0 and 1
        """
        quality_score = 1.0
        
        # Check if response is empty or too short
        if not response or len(response.strip()) < 10:
            quality_score *= 0.1
        
        # Check if response contains citations
        citation_count = response.count('[') + response.count(']')
        if citation_count == 0 and len(context) > 0:
            quality_score *= 0.7  # Penalize lack of citations when context available
        
        # Check for repetition
        words = response.split()
        if len(words) > 10:
            unique_words = len(set(words))
            repetition_ratio = unique_words / len(words)
            if repetition_ratio < 0.5:
                quality_score *= 0.6
        
        # Check if response addresses the query
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words.intersection(response_words))
        if overlap == 0 and len(query_words) > 0:
            quality_score *= 0.8
        
        return min(quality_score, 1.0)
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self._model_loaded = False
            logger.info("Model unloaded successfully")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'model') and self.model is not None:
            self.unload_model()