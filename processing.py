# processing.py

import os
import io
import zipfile
import tempfile
import time
import logging
import json
import traceback
from typing import List, Dict, Any
import asyncio

import pandas as pd
import httpx
import shutil

import config
import prompts
import utils

logger = logging.getLogger(__name__)
number_of_json_retries = config.JSON_RETRIES

async def process_single_document_async(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    file_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Asynchronously processes one document, respecting the concurrency semaphore.
    Includes a retry mechanism for transient JSON parsing errors.
    """
    async with semaphore:
        file_path = file_info['path']
        filename = os.path.basename(file_path)
        logger.info(f"Starting processing for: {filename}")
        extraction_prompt = prompts.get_extraction_prompt()

        # Retry loop for JSON parsing errors
        for attempt in range(number_of_json_retries):
            try:
                response_text = await utils.call_extraction_api_async_with_retry(
                    client, extraction_prompt, file_info['data'], file_info['type']
                )
                json_text = utils.extract_json_from_text(response_text)
                result = json.loads(json_text)

                if 'extracted_fields' in result and isinstance(result['extracted_fields'], list):
                    for field in result['extracted_fields']:
                        if field.get('field_name') == 'date':
                            field['value'] = utils.parse_and_format_date(field.get('value'))
                        elif field.get('field_name') == 'amount_numeric':
                            field['value'] = utils.sanitize_amount(field.get('value'))
                
                result['file_path'] = file_path
                logger.info(f"Successfully processed: {filename}")
                return result

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed for {filename} on attempt {attempt + 1}/{number_of_json_retries}. Retrying...")
                if attempt < 2:
                    await asyncio.sleep(1) # Wait 1 second before retrying API call
                else:
                    logger.error(f"JSON parsing failed for {filename} after 3 attempts. Error: {e}")
                    return {"error": f"JSON Decode Error after retries: {e}", "file_path": file_path}
            
            except Exception as e:
                logger.error(f"An unexpected error occurred while processing {filename}: {e}", exc_info=True)
                return {"error": str(e), "file_path": file_path}
        
        return {"error": "Processing failed after all retries.", "file_path": file_path}


async def process_zip_file_and_generate_report(job_id: str, file_contents: List[bytes], file_names: List[str], job_status_dict: Dict):
    """Orchestrates the entire async extraction and reporting process."""
    job_start_time = time.time()
    try:
        with tempfile.TemporaryDirectory(prefix=f"job_{job_id}_") as temp_dir:
            all_files_to_process = []
            mime_types = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.tiff': 'image/tiff', '.tif': 'image/tiff'}

            for zip_content in file_contents:
                with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
                    for info in zf.infolist():
                        if not info.is_dir() and not info.filename.startswith('__MACOSX'):
                            ext = os.path.splitext(info.filename)[1].lower()
                            if ext in mime_types:
                                all_files_to_process.append({
                                    'path': info.filename,
                                    'data': zf.read(info.filename),
                                    'type': mime_types[ext]
                                })
            
            total_files = len(all_files_to_process)
            job_status_dict.update({"status": "processing", "total_files": total_files, "processed_files": 0, "progress_percentage": 0.0})
            
            semaphore = asyncio.Semaphore(config.API_CONCURRENCY_LIMIT)
            async with httpx.AsyncClient(verify=False) as client:
                tasks = []
                for file_info in all_files_to_process:
                    task = process_single_document_async(client, semaphore, file_info)
                    tasks.append(task)
                
                all_results = await asyncio.gather(*tasks)

            all_data_for_df = []
            for result in all_results:
                if not result or 'file_path' not in result: continue
                folder_name = os.path.basename(os.path.dirname(result['file_path'])) or "root"
                row = {'folder_name': folder_name, 'filepath': os.path.basename(result['file_path'])}
                if "error" in result: row['error'] = result['error']
                for field in result.get('extracted_fields', []):
                    fname = field.get('field_name')
                    if fname:
                        row[fname] = field.get('value')
                        row[f'{fname}_conf'] = field.get('confidence')
                        if field.get('reason'): row[f'{fname}_reason'] = field.get('reason')
                all_data_for_df.append(row)

            df = pd.DataFrame(all_data_for_df)
            base_cols = ['folder_name', 'filepath']
            field_cols = []
            for field_cfg in config.FIELDS:
                fname = field_cfg["name"]
                if fname in df.columns:
                    field_cols.append(fname)
                    if f'{fname}_conf' in df.columns: field_cols.append(f'{fname}_conf')
                    if f'{fname}_reason' in df.columns: field_cols.append(f'{fname}_reason')
            error_col = ['error'] if 'error' in df.columns else []
            final_columns = base_cols + field_cols + error_col
            df = df.reindex(columns=[col for col in final_columns if col in df.columns])

            persistent_csv_path = os.path.join(config.OUTPUT_DIR, f"cheque_extraction_results_{job_id}.csv")
            df.to_csv(persistent_csv_path, index=False, encoding='utf-8')
            logger.info(f"Moved final CSV report to persistent storage: {persistent_csv_path}")

            job_status_dict.update({
                "status": "completed",
                "output_file_path": persistent_csv_path,
                "end_time": time.time(),
                "processing_time": time.time() - job_start_time,
            })
            logger.info(f"Job {job_id} completed. Report at {persistent_csv_path}")

    except Exception as e:
        logger.critical(f"Critical error in job {job_id}: {e}", exc_info=True)
        job_status_dict.update({"status": "failed", "error_message": str(e), "end_time": time.time()})