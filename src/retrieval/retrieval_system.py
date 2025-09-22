"""
Unified retrieval system for multimodal content.

Integrates vector storage, embedding generation, and document processing
to provide fast, accurate retrieval of text, image, and audio content.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import time

from ..models import (
    ContentChunk, ContentType, RetrievalResult, 
    DocumentContent, ProcessingResult, GroundedResponse, Citation
)
from ..config import SystemConfig
from ..processors.router import DocumentRouter
from ..embeddings.unified_embedding_generator import UnifiedEmbeddingGenerator
from .vectordb import QdrantVectorStore


logger = logging.getLogger(__name__)


class MultimodalRetrievalSystem:
    """
    Unified retrieval system for multimodal content.
    
    Features:
    - Document processing and indexing
    - Text and image embedding generation
    - Fast vector similarity search
    - Cross-modal retrieval
    - Response generation with citations
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the multimodal retrieval system.
        
        Args:
            config: System configuration
        """
        self.config = config
        
        # Initialize components
        self.document_router = DocumentRouter(config.processing)
        self.embedding_generator = UnifiedEmbeddingGenerator(config.embedding)
        self.vector_store = QdrantVectorStore(config.storage, config.embedding)
        
        # Load embedding model
        self.embedding_generator.load_model()
        
        logger.info("MultimodalRetrievalSystem initialized successfully")
    
    def add_document(self, file_path: str, document_id: Optional[str] = None) -> ProcessingResult:
        """
        Process and add a document to the retrieval system.
        
        Args:
            file_path: Path to the document file
            document_id: Optional document ID (generated if not provided)
            
        Returns:
            ProcessingResult with success status and details
        """
        try:
            # Route document to appropriate processor
            processor = self.document_router.route_document(file_path)
            
            # Process document
            processing_result = processor.process_document(file_path, document_id)
            
            if processing_result.success and processing_result.document_content:
                # Add chunks to vector store
                self.vector_store.add_chunks(processing_result.document_content.chunks)
                
                logger.info(
                    f"Successfully added document {processing_result.document_content.document_id} "
                    f"with {processing_result.chunks_created} chunks"
                )
            
            return processing_result
            
        except Exception as e:
            error_msg = f"Error adding document {file_path}: {str(e)}"
            logger.error(error_msg)
            return ProcessingResult(
                success=False,
                error_message=error_msg
            )
    
    def add_documents_batch(self, file_paths: List[str]) -> Dict[str, ProcessingResult]:
        """
        Process and add multiple documents in batch.
        
        Args:
            file_paths: List of document file paths
            
        Returns:
            Dictionary mapping file paths to processing results
        """
        results = {}
        
        for file_path in file_paths:
            try:
                results[file_path] = self.add_document(file_path)
            except Exception as e:
                results[file_path] = ProcessingResult(
                    success=False,
                    error_message=f"Batch processing error: {str(e)}"
                )
        
        # Log summary
        successful = sum(1 for r in results.values() if r.success)
        logger.info(f"Batch processing completed: {successful}/{len(file_paths)} documents successful")
        
        return results
    
    def search(
        self, 
        query: str, 
        top_k: int = 10,
        content_types: Optional[List[ContentType]] = None,
        similarity_threshold: float = 0.5,
        include_images: bool = True
    ) -> List[RetrievalResult]:
        """
        Search for relevant content using text query.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            content_types: Filter by content types (None for all)
            similarity_threshold: Minimum similarity score
            include_images: Whether to include image results in cross-modal search
            
        Returns:
            List of retrieval results
        """
        try:
            start_time = time.time()
            
            if include_images and (content_types is None or ContentType.IMAGE in content_types):
                # Cross-modal search
                results = self.vector_store.search_cross_modal(
                    query=query,
                    top_k=top_k,
                    include_images=include_images,
                    similarity_threshold=similarity_threshold
                )
            else:
                # Text-only search
                results = self.vector_store.search(
                    query=query,
                    top_k=top_k,
                    content_types=content_types,
                    similarity_threshold=similarity_threshold
                )
            
            search_time = time.time() - start_time
            logger.info(f"Search completed in {search_time:.3f}s, returned {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in search: {e}")
            raise
    
    def search_by_image(
        self, 
        image_path: str, 
        top_k: int = 10,
        similarity_threshold: float = 0.5
    ) -> List[RetrievalResult]:
        """
        Search for relevant content using image similarity.
        
        Args:
            image_path: Path to query image
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of retrieval results
        """
        try:
            start_time = time.time()
            
            results = self.vector_store.search_by_image(
                image_path=image_path,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            
            search_time = time.time() - start_time
            logger.info(f"Image search completed in {search_time:.3f}s, returned {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in image search: {e}")
            raise
    
    def generate_response(
        self, 
        query: str, 
        context_results: List[RetrievalResult],
        max_length: int = 500,
        temperature: float = 0.7
    ) -> GroundedResponse:
        """
        Generate a grounded response with citations.
        
        Args:
            query: User query
            context_results: Retrieved context for response generation
            max_length: Maximum response length
            temperature: LLM temperature
            
        Returns:
            GroundedResponse with citations
        """
        try:
            # For now, create a simple response with citations
            # In a full implementation, this would use an LLM
            
            if not context_results:
                return GroundedResponse(
                    response_text="I couldn't find relevant information to answer your question.",
                    citations=[],
                    confidence_score=0.0,
                    retrieval_results=[],
                    query=query
                )
            
            # Create citations from retrieval results
            citations = []
            for i, result in enumerate(context_results[:5]):  # Limit to top 5
                citation = Citation(
                    citation_id=i + 1,
                    source_file=result.source_location.file_path,
                    location=result.source_location,
                    excerpt=result.content[:200] + "..." if len(result.content) > 200 else result.content,
                    relevance_score=result.relevance_score,
                    content_type=result.content_type
                )
                citations.append(citation)
            
            # Generate simple response (placeholder for LLM integration)
            response_text = self._generate_simple_response(query, context_results)
            
            confidence_score = min(1.0, sum(r.relevance_score for r in context_results[:3]) / 3)
            
            return GroundedResponse(
                response_text=response_text,
                citations=citations,
                confidence_score=confidence_score,
                retrieval_results=context_results,
                query=query,
                generation_metadata={
                    "model": "simple_retrieval",
                    "temperature": temperature,
                    "max_length": max_length
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def _generate_simple_response(self, query: str, results: List[RetrievalResult]) -> str:
        """Generate a simple response based on retrieved results."""
        if not results:
            return "I couldn't find relevant information to answer your question."
        
        # Combine top results
        context = "\n\n".join([result.content for result in results[:3]])
        
        # Simple template response
        response = f"Based on the available information:\n\n{context}\n\n"
        
        if len(results) > 1:
            response += f"[Sources: {len(results)} relevant documents found]"
        
        return response
    
    def query_with_response(
        self, 
        query: str, 
        top_k: int = 5,
        include_images: bool = True,
        generate_response: bool = True
    ) -> Dict[str, Any]:
        """
        Complete query pipeline: search + optional response generation.
        
        Args:
            query: User query
            top_k: Number of results to retrieve
            include_images: Whether to include image results
            generate_response: Whether to generate a response
            
        Returns:
            Dictionary with search results and optional response
        """
        try:
            # Search for relevant content
            search_results = self.search(
                query=query,
                top_k=top_k,
                include_images=include_images
            )
            
            result = {
                "query": query,
                "search_results": [
                    {
                        "content": r.content,
                        "similarity_score": r.similarity_score,
                        "content_type": r.content_type.value,
                        "source_file": r.source_location.file_path,
                        "metadata": r.metadata
                    }
                    for r in search_results
                ],
                "total_results": len(search_results)
            }
            
            # Generate response if requested
            if generate_response and search_results:
                response = self.generate_response(query, search_results)
                result["response"] = {
                    "text": response.response_text,
                    "confidence_score": response.confidence_score,
                    "citations": [
                        {
                            "id": c.citation_id,
                            "source": c.source_file,
                            "excerpt": c.excerpt,
                            "relevance_score": c.relevance_score
                        }
                        for c in response.citations
                    ]
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in query pipeline: {e}")
            raise
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        try:
            vector_stats = self.vector_store.get_statistics()
            embedding_stats = self.embedding_generator.get_embedding_stats()
            
            return {
                "vector_store": vector_stats,
                "embedding_generator": embedding_stats,
                "supported_formats": self.document_router.get_supported_formats(),
                "configuration": {
                    "chunk_size": self.config.processing.chunk_size,
                    "embedding_dimension": self.config.embedding.embedding_dimension,
                    "text_model": self.config.embedding.text_model_name,
                    "image_model": self.config.embedding.image_model_name
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {}
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks from the system.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            self.vector_store.delete_document(document_id)
            logger.info(f"Successfully deleted document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
    def clear_all_data(self) -> None:
        """Clear all data from the retrieval system."""
        try:
            self.vector_store.clear_collection()
            logger.info("Cleared all data from retrieval system")
            
        except Exception as e:
            logger.error(f"Error clearing data: {e}")
            raise



# New interactive terminal mode for retrieval system
def interactive_terminal_retrieval():
    """
    Runs the retrieval system in interactive terminal mode.
    Loads embeddings from Qdrant and lets the user ask questions.
    """
    from ..config import SystemConfig

    config = SystemConfig()
    retrieval_system = MultimodalRetrievalSystem(config)

    print("\n=== Multimodal Retrieval System (Terminal Mode) ===")
    print("Type your question and press Enter. Type 'exit' to quit.\n")

    while True:
        user_query = input("Your question: ").strip()
        if user_query.lower() in ("exit", "quit"): 
            print("Exiting retrieval system.")
            break
        if not user_query:
            continue

        print("\nSearching for relevant information...")
        try:
            results = retrieval_system.query_with_response(user_query, top_k=5)
            if results["search_results"]:
                print("\nTop Results:")
                for i, result in enumerate(results["search_results"]):
                    print(f"  {i+1}. {result['content'][:120]} (score: {result['similarity_score']:.3f})")
            else:
                print("No relevant results found.")

            if "response" in results:
                print("\nGenerated Answer:")
                print(results["response"]["text"])
                print(f"Confidence: {results['response']['confidence_score']:.2f}")
                if results["response"]["citations"]:
                    print("Citations:")
                    for c in results["response"]["citations"]:
                        print(f"  [{c['id']}] {c['source']} (score: {c['relevance_score']:.2f})")
            print("\n---\n")
        except Exception as e:
            print(f"Error during retrieval: {e}")


if __name__ == "__main__":
    interactive_terminal_retrieval()
