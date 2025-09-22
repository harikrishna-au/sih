"""
Example usage of SemanticRetriever with different search modes.

Demonstrates various search capabilities including text search, cross-modal search,
filtered search, batch search, and reranking functionality.
"""

from ..models import RetrievalResult, SourceLocation, ContentType
from .search_config import SearchConfig, SearchFilter
from .semantic_retriever import SemanticRetriever


def create_semantic_retriever_example():
    """
    Example usage of SemanticRetriever with different search modes.
    """
    from ..config import SystemConfig
    from ..embeddings.unified_embedding_generator import UnifiedEmbeddingGenerator
    from .vectordb.qdrant_vector_store import QdrantVectorStore
    
    # Initialize components
    config = SystemConfig()
    embedding_generator = UnifiedEmbeddingGenerator(config.embedding)
    vector_store = QdrantVectorStore(config.storage, config.embedding)
    
    # Create semantic retriever with reranking enabled
    search_config = SearchConfig(
        similarity_threshold=0.6,
        max_results=50,
        enable_cross_modal=True,
        enable_reranking=True,
        diversity_threshold=0.8,
        reranking_model="hybrid"
    )
    
    retriever = SemanticRetriever(
        vector_store=vector_store,
        embedding_generator=embedding_generator,
        config=search_config
    )
    
    # Example searches
    print("🔎 Semantic Search Examples:")
    
    # Text search
    print("\n1. Text Search:")
    results = retriever.search("machine learning algorithms", top_k=5)
    for i, result in enumerate(results):
        print(f"   {i+1}. {result.content[:100]}... (score: {result.similarity_score:.3f})")
    
    # Cross-modal search
    print("\n2. Cross-modal Search:")
    cross_results = retriever.cross_modal_search("data visualization", top_k=5)
    for i, result in enumerate(cross_results):
        print(f"   {i+1}. [{result.content_type.value}] {result.content[:80]}... (score: {result.similarity_score:.3f})")
    
    # Filtered search
    print("\n3. Filtered Search:")
    search_filter = SearchFilter(
        content_types=[ContentType.TEXT],
        confidence_threshold=0.7
    )
    filtered_results = retriever.search("artificial intelligence", top_k=3, search_filter=search_filter)
    for i, result in enumerate(filtered_results):
        print(f"   {i+1}. {result.content[:100]}... (score: {result.similarity_score:.3f})")
    
    # Batch search
    print("\n4. Batch Search:")
    queries = ["neural networks", "computer vision", "natural language processing"]
    batch_results = retriever.batch_search(queries, top_k=2)
    for i, (query, results) in enumerate(zip(queries, batch_results)):
        print(f"   Query {i+1}: {query}")
        for j, result in enumerate(results):
            print(f"      {j+1}. {result.content[:80]}... (score: {result.similarity_score:.3f})")
    
    # Test reranking functionality
    print("\n5. Reranking Example:")
    sample_results = [
        RetrievalResult(
            chunk_id="chunk_1",
            content="Machine learning algorithms for pattern recognition and data analysis",
            similarity_score=0.8,
            source_location=SourceLocation(file_path="ml_guide.pdf"),
            content_type=ContentType.TEXT,
            metadata={"creation_date": "2023-01-01"}
        ),
        RetrievalResult(
            chunk_id="chunk_2", 
            content="ML techniques",
            similarity_score=0.9,
            source_location=SourceLocation(file_path="short_doc.pdf"),
            content_type=ContentType.TEXT,
            metadata={}
        )
    ]
    
    reranked = retriever.reranker.rerank_results(sample_results, "machine learning algorithms")
    print("   Reranked results:")
    for i, result in enumerate(reranked):
        print(f"      {i+1}. {result.content[:50]}... (relevance: {result.relevance_score:.3f})")
    
    # Get statistics
    stats = retriever.get_search_statistics()
    print(f"\n📊 Search Statistics:")
    print(f"   - Cache size: {stats['cache_size']}")
    print(f"   - Cache hit rate: {stats['hit_rate']:.2%}")


if __name__ == "__main__":
    create_semantic_retriever_example()