"""
Embedding Pipeline for Document Processing

This pipeline handles:
1. PDF chunking
2. Embedding generation 
3. Storage in Qdrant vector DB
4. Retrieval operations

The vector store only accepts pre-generated embeddings.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
import uuid

from ..models import ContentChunk, ContentType, RetrievalResult, SourceLocation, ProcessingResult
from ..config import SystemConfig
from ..processors.router import DocumentRouter
from ..embeddings.unified_embedding_generator import UnifiedEmbeddingGenerator
from ..retrieval.vectordb.faiss_vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    """
    Complete pipeline for processing documents and storing embeddings.
    
    This pipeline:
    1. Processes documents (PDF, DOCX, etc.) into chunks
    2. Generates embeddings for each chunk
    3. Stores embeddings in Qdrant vector DB
    4. Provides retrieval capabilities
    """
    
    def __init__(self, config: SystemConfig):
        """Initialize the embedding pipeline."""
        self.config = config
        
        # Initialize components
        self.document_router = DocumentRouter(config.processing)
        self.embedding_generator = UnifiedEmbeddingGenerator(config.embedding)
        self.vector_store = FAISSVectorStore(
            storage_config=config.storage,
            embedding_dimension=config.embedding.embedding_dimension
        )
        
        # Register processors
        self._register_processors()
        
        # Load embedding model
        self.embedding_generator.load_model()
        
        logger.info("EmbeddingPipeline initialized successfully")
    
    def _register_processors(self):
        """Register document processors."""
        try:
            from ..processors.pdf_processor import PDFProcessor
            self.document_router.register_processor(PDFProcessor, ['pdf'])
            logger.info("Registered PDFProcessor")
        except ImportError:
            logger.warning("PDFProcessor not available")
        
        try:
            from ..processors.docx_processor import DOCXProcessor
            self.document_router.register_processor(DOCXProcessor, ['docx'])
            logger.info("Registered DOCXProcessor")
        except ImportError:
            logger.warning("DOCXProcessor not available")
        
        try:
            from ..processors.text_processor import TextProcessor
            self.document_router.register_processor(TextProcessor, ['txt'])
            logger.info("Registered TextProcessor")
        except ImportError:
            logger.warning("TextProcessor not available")
    
    def process_document(self, file_path: str, document_id: Optional[str] = None) -> ProcessingResult:
        """
        Complete pipeline: Process document → Generate embeddings → Store in vector DB.
        
        Args:
            file_path: Path to the document file
            document_id: Optional document ID
            
        Returns:
            ProcessingResult with success status and details
        """
        try:
            start_time = time.time()
            
            # Step 1: Process document into chunks
            logger.info(f"Step 1: Processing document {file_path}")
            processor = self.document_router.route_document(file_path)
            processing_result = processor.process_document(file_path, document_id)
            
            if not processing_result.success:
                return processing_result
            
            if not processing_result.document_content or not processing_result.document_content.chunks:
                return ProcessingResult(
                    success=False,
                    error_message="No chunks were created from the document"
                )
            
            chunks = processing_result.document_content.chunks
            logger.info(f"Created {len(chunks)} chunks from document")
            
            # Step 2: Generate embeddings for each chunk
            logger.info(f"Step 2: Generating embeddings for {len(chunks)} chunks")
            chunks_with_embeddings = []
            
            for i, chunk in enumerate(chunks):
                try:
                    # Generate embedding based on content type
                    if chunk.content_type == ContentType.TEXT:
                        embedding = self.embedding_generator.encode_text(chunk.content)
                    elif chunk.content_type == ContentType.IMAGE:
                        # For images, we'd need the image path or data
                        # For now, skip or handle differently
                        logger.warning(f"Image embedding not implemented for chunk {chunk.chunk_id}")
                        continue
                    else:
                        logger.warning(f"Unsupported content type {chunk.content_type} for chunk {chunk.chunk_id}")
                        continue
                    
                    # Create new chunk with embedding
                    chunk_with_embedding = ContentChunk(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        content=chunk.content,
                        content_type=chunk.content_type,
                        embedding=embedding,  # Pre-generated embedding
                        source_location=chunk.source_location,
                        metadata=chunk.metadata,
                        confidence_score=chunk.confidence_score
                    )
                    chunks_with_embeddings.append(chunk_with_embedding)
                    
                except Exception as e:
                    logger.error(f"Error generating embedding for chunk {chunk.chunk_id}: {e}")
                    continue
            
            if not chunks_with_embeddings:
                return ProcessingResult(
                    success=False,
                    error_message="No embeddings were generated for any chunks"
                )
            
            logger.info(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")
            
            # Step 3: Store embeddings in vector DB
            logger.info(f"Step 3: Storing {len(chunks_with_embeddings)} embeddings in vector DB")
            self.vector_store.add_chunks(chunks_with_embeddings)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Pipeline completed successfully in {processing_time:.2f}s")
            logger.info(f"Vector DB location: {self.vector_store.get_database_location()}")
            
            return ProcessingResult(
                success=True,
                document_content=processing_result.document_content,
                processing_time=processing_time,
                chunks_created=len(chunks_with_embeddings)
            )
            
        except Exception as e:
            error_msg = f"Pipeline error for {file_path}: {str(e)}"
            logger.error(error_msg)
            return ProcessingResult(
                success=False,
                error_message=error_msg
            )
    
    def search_with_text(
        self, 
        query: str, 
        top_k: int = 10,
        similarity_threshold: float = 0.5
    ) -> List[RetrievalResult]:
        """
        Search the vector DB using a text query.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of retrieval results
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_generator.encode_text(query)
            
            # Search using the pre-generated query embedding
            results = self.vector_store.search_by_vector(
                query_embedding=query_embedding,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            
            logger.info(f"Search returned {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error in search: {e}")
            raise
    
    def get_database_location(self) -> str:
        """Get the location of the vector database."""
        return self.vector_store.get_database_location()
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get comprehensive database information."""
        return self.vector_store.get_database_info()
    
    def load_embeddings_from_file(self, embeddings_file_path: str) -> None:
        """
        Load pre-generated embeddings from an external file.
        
        This is the main use case for the vector store - accepting embeddings
        that were generated elsewhere.
        
        Args:
            embeddings_file_path: Path to JSON file with embeddings
        """
        logger.info(f"Loading pre-generated embeddings from {embeddings_file_path}")
        self.vector_store.load_embeddings_from_file(embeddings_file_path)
        logger.info(f"Embeddings stored in vector DB at: {self.get_database_location()}")
    
    def export_embeddings_to_file(self, output_file_path: str, document_id: Optional[str] = None) -> None:
        """
        Export embeddings from the vector DB to a file.
        
        This allows sharing embeddings with other systems.
        
        Args:
            output_file_path: Path to save the embeddings JSON file
            document_id: Optional filter by document ID
        """
        # This would require implementing an export method in the vector store
        # For now, log the intent
        logger.info(f"Export embeddings to {output_file_path} (not implemented yet)")
    
    def clear_database(self) -> None:
        """Clear all data from the vector database."""
        self.vector_store.clear_collection()
        logger.info("Cleared all data from vector database")


def create_pipeline_example():
    """Example usage of the embedding pipeline."""
    from ..config import ConfigManager
    
    # Initialize pipeline
    config_manager = ConfigManager()
    config = config_manager.load_config()
    pipeline = EmbeddingPipeline(config)
    
    print("=== Embedding Pipeline Example ===")
    print(f"Vector DB will be stored at: {pipeline.get_database_location()}")
    
    # Example: Process a document (you would provide a real PDF path)
    # result = pipeline.process_document("example.pdf")
    # if result.success:
    #     print(f"✅ Processed document with {result.chunks_created} chunks")
    # else:
    #     print(f"❌ Failed: {result.error_message}")
    
    # Example: Load pre-generated embeddings
    # pipeline.load_embeddings_from_file("embeddings.json")
    
    # Example: Search
    # results = pipeline.search_with_text("What is machine learning?", top_k=5)
    # for result in results:
    #     print(f"- {result.content[:100]}... (score: {result.similarity_score:.3f})")
    
    # Get database info
    db_info = pipeline.get_database_info()
    print(f"Database info: {db_info}")


if __name__ == "__main__":
    create_pipeline_example()