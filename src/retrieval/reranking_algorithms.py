"""
Reranking algorithms for improving search result relevance.

Implements various reranking strategies including cross-encoder simulation,
BM25 scoring, and hybrid approaches for enhanced result ordering.
"""

import math
import logging
from typing import List, Dict
from ..models import RetrievalResult
from .relevance_scoring import RelevanceScorer
from .search_config import SearchConfig

logger = logging.getLogger(__name__)


class ResultReranker:
    """Implements various reranking algorithms for search results."""
    
    def __init__(self, config: SearchConfig):
        """Initialize reranker with configuration."""
        self.config = config
        self.relevance_scorer = RelevanceScorer(config)
    
    def rerank_results(
        self, 
        results: List[RetrievalResult], 
        query: str,
        reranking_method: str = None
    ) -> List[RetrievalResult]:
        """
        Rerank search results using specified method.
        
        Args:
            results: Initial search results to rerank
            query: Original search query
            reranking_method: Reranking method to use (overrides config)
            
        Returns:
            Reranked list of results with updated relevance scores
        """
        if not results:
            return results
        
        method = reranking_method or self.config.reranking_model
        
        try:
            # Calculate enhanced relevance scores for each result
            for result in results:
                result.relevance_score = self.relevance_scorer.calculate_relevance_score(result, query)
            
            # Apply method-specific reranking
            if method == "cross_encoder":
                return self._cross_encoder_rerank(results, query)
            elif method == "bm25":
                return self._bm25_rerank(results, query)
            elif method == "hybrid":
                return self._hybrid_rerank(results, query)
            else:
                # Default: sort by relevance score
                return sorted(results, key=lambda x: x.relevance_score, reverse=True)
                
        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Returning original results.")
            return results
    
    def _cross_encoder_rerank(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """
        Rerank using cross-encoder approach (simulated).
        
        In a full implementation, this would use a trained cross-encoder model
        to score query-document pairs directly.
        """
        scored_results = []
        
        for result in results:
            # Simulate cross-encoder scoring with multiple factors
            cross_encoder_score = (
                result.similarity_score * 0.7 +  # Base similarity
                self.relevance_scorer.calculate_query_document_interaction(result, query) * 0.3
            )
            
            # Update relevance score with cross-encoder result
            result.relevance_score = (result.relevance_score + cross_encoder_score) / 2
            scored_results.append(result)
        
        return sorted(scored_results, key=lambda x: x.relevance_score, reverse=True)
    
    def _bm25_rerank(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """
        Rerank using BM25 algorithm.
        
        Implements a simplified version of BM25 scoring for reranking.
        """
        if not results or not query:
            return results
        
        # BM25 parameters
        k1 = 1.2
        b = 0.75
        
        # Calculate average document length
        doc_lengths = [len(result.content.split()) for result in results]
        avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1
        
        # Calculate document frequencies
        query_terms = query.lower().split()
        doc_frequencies = {}
        
        for term in query_terms:
            df = sum(1 for result in results if term in result.content.lower())
            doc_frequencies[term] = df
        
        # Calculate BM25 scores
        scored_results = []
        for i, result in enumerate(results):
            doc_length = doc_lengths[i]
            bm25_score = 0.0
            
            for term in query_terms:
                # Term frequency in document
                tf = result.content.lower().count(term)
                
                # Document frequency
                df = doc_frequencies[term]
                
                # Inverse document frequency
                idf = math.log((len(results) - df + 0.5) / (df + 0.5)) if df > 0 else 0
                
                # BM25 component
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
                
                bm25_score += idf * (numerator / denominator) if denominator > 0 else 0
            
            # Combine with original relevance score
            result.relevance_score = (result.relevance_score + bm25_score / len(query_terms)) / 2
            scored_results.append(result)
        
        return sorted(scored_results, key=lambda x: x.relevance_score, reverse=True)
    
    def _hybrid_rerank(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """
        Hybrid reranking combining multiple approaches.
        
        Combines cross-encoder and BM25 approaches with weighted averaging.
        """
        # Apply cross-encoder reranking
        cross_encoder_results = self._cross_encoder_rerank(results.copy(), query)
        
        # Apply BM25 reranking
        bm25_results = self._bm25_rerank(results.copy(), query)
        
        # Create mapping for efficient lookup
        cross_encoder_scores = {r.chunk_id: r.relevance_score for r in cross_encoder_results}
        bm25_scores = {r.chunk_id: r.relevance_score for r in bm25_results}
        
        # Combine scores with weighted average
        hybrid_results = []
        for result in results:
            ce_score = cross_encoder_scores.get(result.chunk_id, result.relevance_score)
            bm25_score = bm25_scores.get(result.chunk_id, result.relevance_score)
            
            # Weighted combination (60% cross-encoder, 40% BM25)
            result.relevance_score = 0.6 * ce_score + 0.4 * bm25_score
            hybrid_results.append(result)
        
        return sorted(hybrid_results, key=lambda x: x.relevance_score, reverse=True)