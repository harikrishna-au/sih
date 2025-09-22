"""
Document processing API endpoints.

Handles file upload, validation, processing, and batch operations
for multimodal documents.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends, status
from fastapi.responses import JSONResponse
from typing import List, Optional
import uuid
import time
import tempfile
import os
import asyncio
from pathlib import Path

from ...models import ProcessingResult, BatchProcessingStatus, ValidationResult
from ..schemas import (
    DocumentUploadResponse, BatchProcessingResponse, ProcessingStatusResponse,
    FileValidationResponse, ErrorResponse, ProcessingStatus
)
from ..dependencies import (
    get_config, get_document_processor, get_batch_processor, require_api_key
)
from ...monitoring.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Global storage for tracking processing jobs
processing_jobs: dict = {}

async def save_uploaded_file(upload_file: UploadFile, temp_dir: str) -> str:
    """Save uploaded file to temporary directory."""
    file_path = os.path.join(temp_dir, upload_file.filename)
    
    with open(file_path, "wb") as buffer:
        content = await upload_file.read()
        buffer.write(content)
    
    return file_path

async def process_single_file_background(
    file_path: str, 
    job_id: str, 
    filename: str,
    document_processor
):
    """Background task for processing a single file."""
    try:
        logger.info(f"Starting processing for job {job_id}: {filename}")
        
        # Update job status
        if job_id in processing_jobs:
            processing_jobs[job_id].status = ProcessingStatus.PROCESSING.value
            processing_jobs[job_id].current_file = filename
        
        # Process the file
        result = await document_processor.process_file(file_path)
        
        # Update job status based on result
        if job_id in processing_jobs:
            if result.success:
                processing_jobs[job_id].status = ProcessingStatus.COMPLETED.value
                processing_jobs[job_id].processed_files = 1
                logger.info(f"Successfully processed job {job_id}: {filename}")
            else:
                processing_jobs[job_id].status = ProcessingStatus.FAILED.value
                processing_jobs[job_id].failed_files = 1
                processing_jobs[job_id].errors.append(result.error_message or "Unknown error")
                logger.error(f"Failed to process job {job_id}: {result.error_message}")
    
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        if job_id in processing_jobs:
            processing_jobs[job_id].status = ProcessingStatus.FAILED.value
            processing_jobs[job_id].failed_files = 1
            processing_jobs[job_id].errors.append(str(e))
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(file_path)
        except OSError:
            pass

async def process_batch_files_background(
    file_paths: List[str], 
    job_id: str, 
    filenames: List[str],
    batch_processor
):
    """Background task for processing multiple files."""
    try:
        logger.info(f"Starting batch processing for job {job_id}: {len(file_paths)} files")
        
        # Update job status
        if job_id in processing_jobs:
            processing_jobs[job_id].status = ProcessingStatus.PROCESSING.value
        
        # Process files in batch
        status = await batch_processor.process_files(file_paths, job_id)
        
        # Update job status
        if job_id in processing_jobs:
            processing_jobs[job_id].processed_files = status.processed_files
            processing_jobs[job_id].failed_files = status.failed_files
            processing_jobs[job_id].errors = status.errors
            
            if status.failed_files == 0:
                processing_jobs[job_id].status = ProcessingStatus.COMPLETED.value
                logger.info(f"Successfully completed batch job {job_id}")
            else:
                processing_jobs[job_id].status = ProcessingStatus.FAILED.value
                logger.warning(f"Batch job {job_id} completed with {status.failed_files} failures")
    
    except Exception as e:
        logger.error(f"Error in batch processing job {job_id}: {str(e)}")
        if job_id in processing_jobs:
            processing_jobs[job_id].status = ProcessingStatus.FAILED.value
            processing_jobs[job_id].errors.append(str(e))
    
    finally:
        # Clean up temporary files
        for file_path in file_paths:
            try:
                os.unlink(file_path)
            except OSError:
                pass

@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload and process a single document",
    description="Upload a single document for processing. Returns immediately with a job ID for tracking progress."
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file to upload"),
    config=Depends(get_config),
    document_processor=Depends(get_document_processor),
    _=Depends(require_api_key)
):
    """Upload and process a single document."""
    
    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided"
        )
    
    # Check file size
    file_size = 0
    content = await file.read()
    file_size = len(content)
    await file.seek(0)  # Reset file pointer
    
    max_size_bytes = config.processing.max_file_size_mb * 1024 * 1024
    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({file_size} bytes) exceeds maximum allowed size ({max_size_bytes} bytes)"
        )
    
    # Check file format
    file_extension = Path(file.filename).suffix.lower().lstrip('.')
    if file_extension not in config.processing.supported_formats:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file format: {file_extension}. Supported formats: {config.processing.supported_formats}"
        )
    
    # Generate job ID and save file
    job_id = str(uuid.uuid4())
    temp_dir = config.processing.temp_directory
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        file_path = await save_uploaded_file(file, temp_dir)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file"
        )
    
    # Initialize job tracking
    processing_jobs[job_id] = BatchProcessingStatus(
        total_files=1,
        processed_files=0,
        failed_files=0,
        current_file=file.filename,
        start_time=time.time(),
        status=ProcessingStatus.PENDING.value
    )
    
    # Start background processing
    background_tasks.add_task(
        process_single_file_background,
        file_path,
        job_id,
        file.filename,
        document_processor
    )
    
    return DocumentUploadResponse(
        job_id=job_id,
        filename=file.filename,
        file_size=file_size,
        status=ProcessingStatus.PENDING,
        message="File uploaded successfully, processing started",
        estimated_completion_time=time.time() + 30  # Rough estimate
    )

@router.post(
    "/batch-upload",
    response_model=BatchProcessingResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload and process multiple documents",
    description="Upload multiple documents for batch processing. Returns immediately with a job ID for tracking progress."
)
async def batch_upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="List of document files to upload"),
    config=Depends(get_config),
    batch_processor=Depends(get_batch_processor),
    _=Depends(require_api_key)
):
    """Upload and process multiple documents in batch."""
    
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided"
        )
    
    if len(files) > config.processing.batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many files. Maximum batch size is {config.processing.batch_size}"
        )
    
    # Validate all files first
    max_size_bytes = config.processing.max_file_size_mb * 1024 * 1024
    valid_files = []
    file_paths = []
    filenames = []
    
    temp_dir = config.processing.temp_directory
    os.makedirs(temp_dir, exist_ok=True)
    
    for file in files:
        if not file.filename:
            continue
        
        # Check file size
        content = await file.read()
        file_size = len(content)
        await file.seek(0)
        
        if file_size > max_size_bytes:
            logger.warning(f"Skipping file {file.filename}: too large ({file_size} bytes)")
            continue
        
        # Check file format
        file_extension = Path(file.filename).suffix.lower().lstrip('.')
        if file_extension not in config.processing.supported_formats:
            logger.warning(f"Skipping file {file.filename}: unsupported format ({file_extension})")
            continue
        
        valid_files.append(file)
        filenames.append(file.filename)
    
    if not valid_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid files found for processing"
        )
    
    # Save all valid files
    try:
        for file in valid_files:
            file_path = await save_uploaded_file(file, temp_dir)
            file_paths.append(file_path)
    except Exception as e:
        # Clean up any saved files
        for path in file_paths:
            try:
                os.unlink(path)
            except OSError:
                pass
        
        logger.error(f"Failed to save uploaded files: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded files"
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job tracking
    processing_jobs[job_id] = BatchProcessingStatus(
        total_files=len(valid_files),
        processed_files=0,
        failed_files=0,
        start_time=time.time(),
        status=ProcessingStatus.PENDING.value
    )
    
    # Start background processing
    background_tasks.add_task(
        process_batch_files_background,
        file_paths,
        job_id,
        filenames,
        batch_processor
    )
    
    return BatchProcessingResponse(
        job_id=job_id,
        total_files=len(valid_files),
        status=ProcessingStatus.PENDING,
        message=f"Batch upload successful, processing {len(valid_files)} files",
        estimated_completion_time=time.time() + (len(valid_files) * 10)  # Rough estimate
    )

@router.get(
    "/status/{job_id}",
    response_model=ProcessingStatusResponse,
    summary="Get processing job status",
    description="Get the current status of a document processing job."
)
async def get_processing_status(
    job_id: str,
    _=Depends(require_api_key)
):
    """Get the status of a processing job."""
    
    if job_id not in processing_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    job_status = processing_jobs[job_id]
    current_time = time.time()
    
    # Calculate progress percentage
    if job_status.total_files > 0:
        progress_percentage = (job_status.processed_files / job_status.total_files) * 100
    else:
        progress_percentage = 0.0
    
    # Calculate processing time
    processing_time = None
    if job_status.start_time:
        if job_status.status in [ProcessingStatus.COMPLETED.value, ProcessingStatus.FAILED.value]:
            processing_time = current_time - job_status.start_time
        else:
            processing_time = current_time - job_status.start_time
    
    return ProcessingStatusResponse(
        job_id=job_id,
        status=job_status.status,
        total_files=job_status.total_files,
        processed_files=job_status.processed_files,
        failed_files=job_status.failed_files,
        current_file=job_status.current_file,
        start_time=job_status.start_time,
        completion_time=current_time if job_status.status in [ProcessingStatus.COMPLETED.value, ProcessingStatus.FAILED.value] else None,
        processing_time=processing_time,
        errors=job_status.errors,
        progress_percentage=progress_percentage
    )

@router.delete(
    "/jobs/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel or delete a processing job",
    description="Cancel an active processing job or delete a completed job from tracking."
)
async def delete_processing_job(
    job_id: str,
    _=Depends(require_api_key)
):
    """Cancel or delete a processing job."""
    
    if job_id not in processing_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found"
        )
    
    # Remove job from tracking
    del processing_jobs[job_id]
    logger.info(f"Deleted processing job {job_id}")

@router.get(
    "/jobs",
    response_model=List[ProcessingStatusResponse],
    summary="List all processing jobs",
    description="Get a list of all processing jobs and their current status."
)
async def list_processing_jobs(
    status_filter: Optional[ProcessingStatus] = None,
    limit: int = 100,
    _=Depends(require_api_key)
):
    """List all processing jobs with optional status filtering."""
    
    jobs = []
    current_time = time.time()
    
    for job_id, job_status in processing_jobs.items():
        # Apply status filter if provided
        if status_filter and job_status.status != status_filter:
            continue
        
        # Calculate progress percentage
        if job_status.total_files > 0:
            progress_percentage = (job_status.processed_files / job_status.total_files) * 100
        else:
            progress_percentage = 0.0
        
        # Calculate processing time
        processing_time = None
        if job_status.start_time:
            if job_status.status in [ProcessingStatus.COMPLETED.value, ProcessingStatus.FAILED.value]:
                processing_time = current_time - job_status.start_time
            else:
                processing_time = current_time - job_status.start_time
        
        job_response = ProcessingStatusResponse(
            job_id=job_id,
            status=job_status.status,
            total_files=job_status.total_files,
            processed_files=job_status.processed_files,
            failed_files=job_status.failed_files,
            current_file=job_status.current_file,
            start_time=job_status.start_time,
            completion_time=current_time if job_status.status in [ProcessingStatus.COMPLETED.value, ProcessingStatus.FAILED.value] else None,
            processing_time=processing_time,
            errors=job_status.errors,
            progress_percentage=progress_percentage
        )
        
        jobs.append(job_response)
        
        if len(jobs) >= limit:
            break
    
    return jobs

@router.post(
    "/validate",
    response_model=FileValidationResponse,
    summary="Validate a file without processing",
    description="Validate a file to check if it can be processed without actually processing it."
)
async def validate_file(
    file: UploadFile = File(..., description="File to validate"),
    config=Depends(get_config),
    document_processor=Depends(get_document_processor),
    _=Depends(require_api_key)
):
    """Validate a file without processing it."""
    
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided"
        )
    
    # Check file size
    content = await file.read()
    file_size = len(content)
    
    max_size_bytes = config.processing.max_file_size_mb * 1024 * 1024
    warnings = []
    
    if file_size > max_size_bytes:
        return FileValidationResponse(
            filename=file.filename,
            is_valid=False,
            file_size=file_size,
            error_message=f"File size ({file_size} bytes) exceeds maximum allowed size ({max_size_bytes} bytes)"
        )
    
    # Check file format
    file_extension = Path(file.filename).suffix.lower().lstrip('.')
    if file_extension not in config.processing.supported_formats:
        return FileValidationResponse(
            filename=file.filename,
            is_valid=False,
            file_format=file_extension,
            file_size=file_size,
            error_message=f"Unsupported file format: {file_extension}. Supported formats: {config.processing.supported_formats}"
        )
    
    # Add warnings for large files
    if file_size > (max_size_bytes * 0.8):
        warnings.append("File is close to the maximum size limit")
    
    return FileValidationResponse(
        filename=file.filename,
        is_valid=True,
        file_format=file_extension,
        file_size=file_size,
        warnings=warnings
    )