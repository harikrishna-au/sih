"""
Main FastAPI application for the multimodal RAG system.

Provides RESTful API endpoints for document processing, semantic search,
and LLM-powered response generation with citations.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import asyncio
import uuid
import time
from pathlib import Path
import tempfile
import os

from ..config import ConfigManager, SystemConfig
from ..models import (
    ProcessingResult, BatchProcessingStatus, ValidationResult,
    RetrievalResult, GroundedResponse
)
from .schemas import (
    DocumentUploadResponse, BatchProcessingResponse, ProcessingStatusResponse,
    SearchRequest, SearchResponse, QuestionRequest, QuestionResponse,
    ErrorResponse
)
from .dependencies import get_config, get_document_processor, get_batch_processor
from ..monitoring.logger import get_logger

logger = get_logger(__name__)

# Global storage for tracking processing jobs
processing_jobs: Dict[str, BatchProcessingStatus] = {}

def create_app(config: Optional[SystemConfig] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    
    if config is None:
        config_manager = ConfigManager()
        config = config_manager.load_config()
    
    app = FastAPI(
        title="Multimodal RAG System API",
        description="RESTful API for multimodal document processing and semantic retrieval",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Configure CORS
    if config.api.cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.api.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Include routers
    from .routers import documents, search, generation
    app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
    app.include_router(search.router, prefix="/api/v1/search", tags=["search"])
    app.include_router(generation.router, prefix="/api/v1/generate", tags=["generation"])
    

    
    @app.get("/")
    async def root():
        """Root endpoint providing API information."""
        return {
            "name": "Multimodal RAG System API",
            "version": "1.0.0",
            "status": "online",
            "endpoints": {
                "documents": "/api/v1/documents",
                "search": "/api/v1/search", 
                "generation": "/api/v1/generate",
                "docs": "/docs"
            }
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "1.0.0"
        }
    
    return app

# Create the app instance
app = create_app()

if __name__ == "__main__":
    import uvicorn
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    uvicorn.run(
        "src.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug,
        log_level="info"
    )