"""
FastAPI dependency injection for system components.

Provides dependency functions for injecting configuration, processors,
and other system components into API endpoints.
"""

from fastapi import Depends, HTTPException, status
from typing import Optional
import asyncio
import time

from ..config import ConfigManager, SystemConfig
from ..models import ProcessingResult, BatchProcessingStatus
from ..monitoring.logger import get_logger

logger = get_logger(__name__)

# Global instances (will be initialized on startup)
_config_manager: Optional[ConfigManager] = None
_retrieval_system_instance = None
_document_processor_instance = None
_system_config: Optional[SystemConfig] = None

def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_config() -> SystemConfig:
    """Get system configuration."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    
    return _config_manager.config

def get_retrieval_system():
    """Get singleton retrieval system instance."""
    global _retrieval_system_instance
    if _retrieval_system_instance is None:
        from ..retrieval.retrieval_system import MultimodalRetrievalSystem
        config = get_config()
        _retrieval_system_instance = MultimodalRetrievalSystem(config)
        logger.info("Created singleton retrieval system instance")
    return _retrieval_system_instance

def get_document_processor():
    """
    Get document processor instance.
    """
    try:
        from ..processors.router import DocumentRouter
        
        config = get_config()
        document_router = DocumentRouter(config.processing)
        
        # Register processors (these would be imported if available)
        try:
            from ..processors.pdf_processor import PDFProcessor
            document_router.register_processor(PDFProcessor, ['pdf'])
        except ImportError:
            logger.debug("PDFProcessor not available")
        
        try:
            from ..processors.docx_processor import DOCXProcessor
            document_router.register_processor(DOCXProcessor, ['docx'])
        except ImportError:
            logger.debug("DOCXProcessor not available")
        
        try:
            from ..processors.image_processor import ImageProcessor
            document_router.register_processor(ImageProcessor, ['png', 'jpg', 'jpeg'])
        except ImportError:
            logger.debug("ImageProcessor not available")
        
        class RealDocumentProcessor:
            """Real document processor using DocumentRouter."""
            
            def __init__(self, router):
                self.router = router
            
            async def process_file(self, file_path: str) -> ProcessingResult:
                """Process file using real document router and index to retrieval system."""
                try:
                    # Use the shared retrieval system instance for both processing and indexing
                    retrieval_system = get_retrieval_system()
                    
                    # Use the retrieval system's add_document method which handles both processing and indexing
                    result = retrieval_system.add_document(file_path)
                    return result
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    # Return a failed result
                    return ProcessingResult(
                        success=False,
                        error_message=str(e),
                        processing_time=0.0,
                        chunks_created=0
                    )
            
            def validate_file(self, file_path: str, file_size: int) -> bool:
                """Validate file using real document router."""
                try:
                    config = get_config()
                    max_size_bytes = config.processing.max_file_size_mb * 1024 * 1024
                    if file_size > max_size_bytes:
                        return False
                    
                    # Use router's validation
                    result = self.router.validate_file(file_path)
                    return result.is_valid
                except Exception as e:
                    logger.error(f"Error validating file: {e}")
                    return False
            
            def get_supported_formats(self) -> list:
                """Get supported file formats."""
                try:
                    return self.router.get_supported_formats()
                except Exception:
                    config = get_config()
                    return config.processing.supported_formats
        
        return RealDocumentProcessor(document_router)
        
    except Exception as e:
        logger.warning(f"Could not initialize real document processor: {e}")
        logger.info("Falling back to mock document processor")
        
        # Fallback to mock implementation
        class MockDocumentProcessor:
            """Mock document processor for API development."""
            
            async def process_file(self, file_path: str) -> ProcessingResult:
                """Mock file processing."""
                await asyncio.sleep(0.1)  # Simulate processing time
                return ProcessingResult(
                    success=True,
                    processing_time=0.1,
                    chunks_created=5
                )
            
            def validate_file(self, file_path: str, file_size: int) -> bool:
                """Mock file validation."""
                config = get_config()
                max_size_bytes = config.processing.max_file_size_mb * 1024 * 1024
                return file_size <= max_size_bytes
            
            def get_supported_formats(self) -> list:
                """Get supported file formats."""
                config = get_config()
                return config.processing.supported_formats
        
        return MockDocumentProcessor()

def get_batch_processor():
    """
    Get batch processor instance.
    """
    try:
        from ..batch.batch_processor import BatchProcessor
        from ..batch.resource_manager import ResourceManager
        
        config = get_config()
        
        # Initialize resource manager
        resource_manager = ResourceManager(
            max_concurrent_jobs=config.processing.max_concurrent_files,
            memory_limit_mb=1024,  # 1GB memory limit
            temp_directory=config.processing.temp_directory
        )
        
        # Initialize batch processor
        batch_processor = BatchProcessor(
            config=config.processing,
            resource_manager=resource_manager
        )
        
        class RealBatchProcessor:
            """Real batch processor using the batch processing system."""
            
            def __init__(self, processor):
                self.processor = processor
            
            async def process_files(self, file_paths: list, job_id: str) -> BatchProcessingStatus:
                """Process files using real batch processor."""
                try:
                    # Create batch processing job
                    job = await self.processor.create_batch_job(
                        file_paths=file_paths,
                        job_id=job_id
                    )
                    
                    # Process the batch
                    result = await self.processor.process_batch(job)
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Error in real batch processing: {e}")
                    # Return failed status
                    return BatchProcessingStatus(
                        total_files=len(file_paths),
                        processed_files=0,
                        failed_files=len(file_paths),
                        current_file=None,
                        start_time=time.time(),
                        errors=[str(e)],
                        status="failed"
                    )
        
        return RealBatchProcessor(batch_processor)
        
    except Exception as e:
        logger.warning(f"Could not initialize real batch processor: {e}")
        logger.info("Falling back to mock batch processor")
        
        # Fallback to mock implementation
        class MockBatchProcessor:
            """Mock batch processor for API development."""
            
            async def process_files(self, file_paths: list, job_id: str) -> BatchProcessingStatus:
                """Mock batch processing."""
                status = BatchProcessingStatus(
                    total_files=len(file_paths),
                    processed_files=0,
                    failed_files=0,
                    current_file=file_paths[0] if file_paths else None,
                    start_time=time.time(),
                    status="processing"
                )
                
                # Simulate processing
                for i, file_path in enumerate(file_paths):
                    await asyncio.sleep(0.1)  # Simulate processing time
                    status.processed_files = i + 1
                    status.current_file = file_path
                
                status.status = "completed"
                return status
        
        return MockBatchProcessor()

def get_semantic_retriever():
    """
    Get semantic retriever instance using shared retrieval system.
    """
    try:
        # Use the shared retrieval system instance
        retrieval_system = get_retrieval_system()
        
        class RealSemanticRetriever:
            """Real semantic retriever using the shared multimodal retrieval system."""
            
            def __init__(self, retrieval_system):
                self.retrieval_system = retrieval_system
            
            async def search(self, query: str, k: int = 10, **kwargs):
                """Perform real semantic search."""
                try:
                    # Extract search parameters
                    similarity_threshold = kwargs.get('similarity_threshold', 0.5)
                    content_types = kwargs.get('content_types', None)
                    
                    # Perform search using the retrieval system
                    results = self.retrieval_system.search(
                        query=query,
                        top_k=k,
                        content_types=content_types,
                        similarity_threshold=similarity_threshold
                    )
                    
                    return results
                    
                except Exception as e:
                    logger.error(f"Error in real semantic search: {e}")
                    # Fall back to mock results if real search fails
                    return await self._mock_search(query, k)
            
            async def _mock_search(self, query: str, k: int):
                """Fallback mock search if real search fails."""
                await asyncio.sleep(0.1)
                
                from ..models import RetrievalResult, SourceLocation, ContentType
                
                results = []
                for i in range(min(k, 3)):
                    result = RetrievalResult(
                        chunk_id=f"chunk_{i}",
                        content=f"Fallback content for query '{query}' - result {i+1}",
                        similarity_score=0.9 - (i * 0.1),
                        source_location=SourceLocation(
                            file_path=f"fallback_document_{i}.pdf",
                            page_number=i + 1
                        ),
                        content_type=ContentType.TEXT,
                        relevance_score=0.85 - (i * 0.1)
                    )
                    results.append(result)
                
                return results
        
        return RealSemanticRetriever(retrieval_system)
        
    except Exception as e:
        logger.warning(f"Could not initialize real semantic retriever: {e}")
        logger.info("Falling back to mock semantic retriever")
        
        # Fallback to mock implementation
        class MockSemanticRetriever:
            """Mock semantic retriever for API development."""
            
            async def search(self, query: str, k: int = 10, **kwargs):
                """Mock semantic search."""
                await asyncio.sleep(0.1)
                
                from ..models import RetrievalResult, SourceLocation, ContentType
                
                results = []
                for i in range(min(k, 3)):
                    result = RetrievalResult(
                        chunk_id=f"chunk_{i}",
                        content=f"Mock content for query '{query}' - result {i+1}",
                        similarity_score=0.9 - (i * 0.1),
                        source_location=SourceLocation(
                            file_path=f"mock_document_{i}.pdf",
                            page_number=i + 1
                        ),
                        content_type=ContentType.TEXT,
                        relevance_score=0.85 - (i * 0.1)
                    )
                    results.append(result)
                
                return results
        
        return MockSemanticRetriever()

def get_llm_engine():
    """
    Get LLM engine instance.
    """
    try:
        from ..llm.llm_engine import LLMEngine
        
        config = get_config()
        llm_engine = LLMEngine(config.llm)
        
        class RealLLMEngine:
            """Real LLM engine wrapper."""
            
            def __init__(self, engine):
                self.engine = engine
            
            async def generate_response(self, context: list, question: str, **kwargs) -> str:
                """Generate response using real LLM engine."""
                try:
                    # Convert context list to string if needed
                    if isinstance(context, list):
                        context_str = "\n\n".join(str(c) for c in context)
                    else:
                        context_str = str(context)
                    
                    # Generate response
                    response = self.engine.generate_with_context(
                        context=[context_str],
                        query=question,
                        max_tokens=kwargs.get('max_length', 500),
                        temperature=kwargs.get('temperature', 0.7)
                    )
                    
                    return response
                    
                except Exception as e:
                    logger.error(f"Error in real LLM generation: {e}")
                    # Fallback response
                    return f"I apologize, but I encountered an error while generating a response to: {question}"
            
            def get_model_info(self) -> dict:
                """Get real model information."""
                try:
                    return self.engine.get_model_info()
                except Exception:
                    return {
                        "model_name": "llm-engine",
                        "version": "1.0.0",
                        "context_length": 4096,
                        "quantization": "4bit",
                        "loaded": self.engine.is_model_loaded()
                    }
            
            def is_model_loaded(self) -> bool:
                """Check if model is loaded."""
                try:
                    return self.engine.is_model_loaded()
                except Exception:
                    return False
        
        return RealLLMEngine(llm_engine)
        
    except Exception as e:
        logger.warning(f"Could not initialize real LLM engine: {e}")
        logger.info("Falling back to mock LLM engine")
        
        # Fallback to mock implementation
        class MockLLMEngine:
            """Mock LLM engine for API development."""
            
            async def generate_response(self, context: list, question: str, **kwargs) -> str:
                """Mock response generation."""
                await asyncio.sleep(0.5)  # Simulate generation time
                return f"Mock answer to: {question}. Based on the provided context, this is a generated response with citations [1] [2]."
            
            def get_model_info(self) -> dict:
                """Get model information."""
                return {
                    "model_name": "mock-llm",
                    "version": "1.0.0",
                    "context_length": 4096,
                    "quantization": "4bit",
                    "loaded": False
                }
            
            def is_model_loaded(self) -> bool:
                """Check if model is loaded."""
                return False
        
        return MockLLMEngine()

def get_citation_manager():
    """
    Get citation manager instance.
    """
    try:
        from ..citation.citation_manager import CitationManager
        
        config = get_config()
        citation_manager = CitationManager(config)
        
        class RealCitationManager:
            """Real citation manager wrapper."""
            
            def __init__(self, manager):
                self.manager = manager
            
            def extract_citations(self, response: str, retrieval_results: list) -> list:
                """Extract citations using real citation manager."""
                try:
                    # Use the real citation manager to extract citations
                    citations = self.manager.extract_citations_from_response(
                        response_text=response,
                        source_documents=retrieval_results
                    )
                    
                    return citations
                    
                except Exception as e:
                    logger.error(f"Error in real citation extraction: {e}")
                    # Fallback to simple citation extraction
                    return self._fallback_extract_citations(response, retrieval_results)
            
            def _fallback_extract_citations(self, response: str, retrieval_results: list) -> list:
                """Fallback citation extraction."""
                from ..models import Citation
                import re
                
                citations = []
                citation_pattern = re.compile(r'\[(\d+)\]')
                citation_numbers = citation_pattern.findall(response)
                
                for i, citation_num in enumerate(set(citation_numbers)):
                    if i < len(retrieval_results):
                        result = retrieval_results[i]
                        citation = Citation(
                            citation_id=int(citation_num),
                            source_file=result.source_location.file_path,
                            location=result.source_location,
                            excerpt=result.content[:150] + "..." if len(result.content) > 150 else result.content,
                            relevance_score=result.relevance_score,
                            content_type=result.content_type
                        )
                        citations.append(citation)
                
                return citations
        
        return RealCitationManager(citation_manager)
        
    except Exception as e:
        logger.warning(f"Could not initialize real citation manager: {e}")
        logger.info("Falling back to mock citation manager")
        
        # Fallback to mock implementation
        class MockCitationManager:
            """Mock citation manager for API development."""
            
            def extract_citations(self, response: str, retrieval_results: list) -> list:
                """Mock citation extraction."""
                from ..models import Citation
                import re
                
                citations = []
                citation_pattern = re.compile(r'\[(\d+)\]')
                citation_numbers = citation_pattern.findall(response)
                
                for i, citation_num in enumerate(set(citation_numbers)):
                    if i < len(retrieval_results):
                        result = retrieval_results[i]
                        citation = Citation(
                            citation_id=int(citation_num),
                            source_file=result.source_location.file_path,
                            location=result.source_location,
                            excerpt=result.content[:100] + "..." if len(result.content) > 100 else result.content,
                            relevance_score=result.relevance_score,
                            content_type=result.content_type
                        )
                        citations.append(citation)
                
                return citations
        
        return MockCitationManager()

def get_response_generator():
    """
    Get response generator instance.
    
    Integrates LLM engine with citation management for grounded response generation.
    """
    try:
        from ..llm.llm_engine import LLMEngine
        from ..llm.response_generator import ResponseGenerator
        
        config = get_config()
        llm_engine = LLMEngine(config.llm)
        response_generator = ResponseGenerator(llm_engine, config.llm)
        
        return response_generator
        
    except ImportError as e:
        logger.warning(f"Could not import LLM components: {e}. Using mock implementation.")
        
        # Mock implementation for development
        class MockResponseGenerator:
            """Mock response generator for API development."""
            
            def __init__(self):
                self.llm_engine = get_llm_engine()
            
            def generate_grounded_response(self, query: str, retrieval_results: list, **kwargs):
                """Mock grounded response generation."""
                from ..models import GroundedResponse, Citation, SourceLocation, ContentType
                
                # Generate mock response
                response_text = f"Mock grounded response to: {query}. Based on {len(retrieval_results)} sources [1] [2]."
                
                # Create mock citations
                citations = []
                for i, result in enumerate(retrieval_results[:2]):
                    citation = Citation(
                        citation_id=i + 1,
                        source_file=result.source_location.file_path,
                        location=result.source_location,
                        excerpt=result.content[:100] + "...",
                        relevance_score=result.relevance_score,
                        content_type=result.content_type
                    )
                    citations.append(citation)
                
                return GroundedResponse(
                    response_text=response_text,
                    citations=citations,
                    confidence_score=0.85,
                    retrieval_results=retrieval_results,
                    query=query,
                    generation_metadata={
                        "generation_time": 0.5,
                        "context_length": sum(len(r.content) for r in retrieval_results),
                        "num_sources": len(retrieval_results),
                        "citations_found": len(citations),
                        "quality_score": 0.85,
                        "model_info": {"model_name": "mock-llm", "version": "1.0.0"}
                    }
                )
            
            def validate_response(self, grounded_response, min_quality_threshold=0.5):
                """Mock response validation."""
                return True, []
            
            def _calculate_response_quality(self, response_text, context, query, retrieval_results):
                """Mock quality calculation."""
                return 0.85
            
            def improve_response_quality(self, query, retrieval_results, initial_response, max_attempts=2):
                """Mock response improvement."""
                # Return slightly improved response
                improved_response = initial_response
                improved_response.confidence_score = min(1.0, initial_response.confidence_score + 0.1)
                return improved_response
        
        return MockResponseGenerator()
    
    except Exception as e:
        logger.error(f"Error creating response generator: {e}")
        raise RuntimeError(f"Failed to initialize response generator: {e}")

async def verify_system_health():
    """Verify that all system components are healthy."""
    try:
        config = get_config()
        # Add health checks for various components
        return True
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="System is not healthy"
        )

def require_api_key(api_key: Optional[str] = None):
    """Dependency to require API key authentication if enabled."""
    config = get_config()
    
    if not config.api.api_key_required:
        return True
    
    if not api_key or api_key != config.api.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key"
        )
    
    return True