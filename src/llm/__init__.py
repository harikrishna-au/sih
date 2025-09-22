"""
LLM integration module for the multimodal RAG system.

This module provides local LLM integration using quantized models
for response generation with citation support.
"""

from .llm_engine import LLMEngine
from .response_generator import ResponseGenerator

__all__ = ['LLMEngine', 'ResponseGenerator']