"""
Search configuration and filter classes for semantic retrieval.

Defines configuration options, search filters, and search parameters
for the semantic retrieval system.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from ..models import ContentType


@dataclass
class SearchFilter:
    """Filter conditions for semantic search."""
    content_types: Optional[List[ContentType]] = None
    document_ids: Optional[List[str]] = None
    file_paths: Optional[List[str]] = None
    date_range: Optional[Tuple[str, str]] = None
    confidence_threshold: float = 0.0
    metadata_filters: Optional[Dict[str, Any]] = None


@dataclass
class SearchConfig:
    """Configuration for semantic search operations."""
    similarity_threshold: float = 0.5
    max_results: int = 100
    enable_cross_modal: bool = True
    enable_reranking: bool = True
    diversity_threshold: float = 0.8
    query_expansion: bool = False
    reranking_model: str = "cross_encoder"  # Options: "cross_encoder", "bm25", "hybrid"
    relevance_weights: Dict[str, float] = None  # Weights for different relevance factors
    
    def __post_init__(self):
        """Set default relevance weights if not provided."""
        if self.relevance_weights is None:
            self.relevance_weights = {
                "semantic_similarity": 0.6,
                "content_type_boost": 0.1,
                "recency_boost": 0.1,
                "length_penalty": 0.1,
                "keyword_match": 0.1
            }