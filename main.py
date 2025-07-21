# main.py

import logging
import uuid
import time
import os
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import config
import utils
from processing import process_zip_file_and_generate_report

utils.configure_logging()
logger = logging.getLogger(__name__)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

app = FastAPI(
    title="High-Performance Cheque Data Extraction API",
    description="Processes cheque images using a fully asynchronous pipeline with rate limiting.",
    version="5.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processed_jobs: Dict[str, Dict] = {}


@app.post("/upload")
async def upload_files_for_processing(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """Accepts ZIP files and starts the async extraction in the background."""
    job_id = str(uuid.uuid4())
    logger.info(f"Received new job with ID: {job_id}")

    file_contents = []
    file_names = []
    for file in files:
        if not file.filename or not file.filename.lower().endswith('.zip'):
            raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}. Only .zip files are accepted.")
        file_contents.append(await file.read())
        file_names.append(file.filename)

    processed_jobs[job_id] = {
        "job_id": job_id, "status": "queued", "start_time": time.time(),
        "input_files": file_names, "total_files": 0, "processed_files": 0,
        "progress_percentage": 0.0,
    }

    background_tasks.add_task(
        process_zip_file_and_generate_report,
        job_id,
        file_contents,
        file_names,
        processed_jobs[job_id]
    )

    return {
        "message": "Job successfully queued for async processing.",
        "job_id": job_id,
        "status_endpoint": f"/status/{job_id}"
    }


@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Returns the current status of a processing job."""
    job = processed_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.get("/download/{job_id}")
async def download_result_file(job_id: str):
    """Allows downloading of the generated CSV report for a completed job."""
    job = processed_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail=f"Job is not complete. Current status: {job.get('status')}")

    output_path = job.get("output_file_path")
    if not output_path or not os.path.exists(output_path):
        logger.error(f"File not found for job {job_id} at path: {output_path}")
        raise HTTPException(status_code=404, detail="Output file not found on server.")

    return FileResponse(
        path=output_path,
        filename=os.path.basename(output_path),
        media_type="text/csv"
    )


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint for health checks."""
    return {"message": "Cheque Extraction API is running."}