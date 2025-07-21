# utils.py

import logging
import time
import random
import json
import re
import base64
import asyncio
from typing import Optional
from logging.handlers import RotatingFileHandler
import os

import httpx
from dateutil import parser

import config
from schemas import APIRequestBody, APIRequestMessage, APIRequestMessageContent, APIResponseBody

logger = logging.getLogger(__name__)
number_of_api_retries = config.API_RETRIES
exponential_backoff_factor = config.EXPONENTIAL_BACKOFF_FACTOR

def parse_and_format_date(date_str: Optional[str]) -> Optional[str]:
    """Parses a date string and returns it as YYYY-MM-DD."""
    if not date_str or not isinstance(date_str, str):
        return date_str
    try:
        parsed_date = parser.parse(date_str, dayfirst=True)
        return parsed_date.strftime('%Y-%m-%d')
    except (parser.ParserError, TypeError):
        logger.warning(f"Could not parse date: '{date_str}'. Returning original.")
        return date_str


def sanitize_amount(amount_str: Optional[str]) -> Optional[str]:
    """Cleans an amount string to be a valid number representation."""
    if not amount_str or not isinstance(amount_str, str):
        return amount_str
    cleaned_str = re.sub(r'[^\d.]', '', amount_str)
    if cleaned_str.count('.') > 1:
        parts = cleaned_str.split('.')
        cleaned_str = "".join(parts[:-1]) + "." + parts[-1]
    return cleaned_str


def configure_logging():
    """
    Configures logging to write to a rotating file and the console.
    """
    # Ensure the log directory exists
    os.makedirs(config.LOG_DIR, exist_ok=True)
    log_file_path = os.path.join(config.LOG_DIR, config.LOG_FILENAME)

    # Define log message format
    log_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if this function is called multiple times
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # --- Rotating File Handler ---
    # Rotates logs when they reach 10MB, keeping 5 old log files as backup.
    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    # --- Console (Stream) Handler ---
    # This keeps printing logs to your terminal.
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(stream_handler)

    logger.info("Logging configured to write to file and console.")


def extract_json_from_text(text: str) -> str:
    """Extracts a JSON object from a string, including from markdown blocks."""
    match = re.search(r'```json\s*(\{[\s\S]*\})\s*```', text)
    if match:
        return match.group(1).strip()
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end+1].strip()
    logger.warning("Could not find clear JSON markers, returning raw text.")
    return text

async def call_extraction_api_async_with_retry(
    client: httpx.AsyncClient,
    prompt: str,
    image_data: bytes,
    mime_type: str,
    max_retries: int = number_of_api_retries,
    initial_delay: float = 1.5
) -> str:
    """
    Calls the external LLM API asynchronously with robust retry logic and exponential backoff.
    Args:
        client: An httpx.AsyncClient instance for connection pooling.
        prompt: The text prompt for the model.
        image_data: The image file in bytes.
        mime_type: The MIME type of the image.
        max_retries: Maximum number of retries for network errors.
        initial_delay: Initial delay in seconds for retries.
    """
    headers = {'Content-Type': 'application/json', 'x-goog-api-key': config.API_KEY}
    base64_image = base64.b64encode(image_data).decode('utf-8')

    messages = [
        APIRequestMessage(role="user", content=[
            APIRequestMessageContent(type="text", text=prompt),
            APIRequestMessageContent(type="image_url", image_url={"url": f"data:{mime_type};base64,{base64_image}"})
        ])
    ]

    api_request_body = APIRequestBody(model=config.MODEL_NAME, messages=messages)
    data = api_request_body.model_dump(exclude_none=True)

    delay = initial_delay
    for i in range(max_retries):
        try:
            start_time = time.monotonic()
            response = await client.post(config.API_URL, headers=headers, json=data, timeout=300.0)
            duration = time.monotonic() - start_time
            logger.info(f"API call to {config.API_URL} responded in {duration:.2f} seconds.")

            response.raise_for_status()
            api_response = APIResponseBody.model_validate(response.json())

            if api_response.choices and api_response.choices[0].message.content:
                return api_response.choices[0].message.content
            raise ValueError("API response is valid but missing expected content.")

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            is_server_error = isinstance(e, httpx.HTTPStatusError) and e.response.status_code >= 500
            is_rate_limit_error = isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429
            is_network_error = isinstance(e, httpx.RequestError)

            # Only retry on server errors, rate limits, or network errors.
            if not (is_server_error or is_rate_limit_error or is_network_error):
                logger.error(f"Non-retryable client error: {e}")
                raise

            logger.warning(f"API call failed (Attempt {i + 1}/{max_retries}): {e}.")
            if i == max_retries - 1:
                logger.error("Max retries exceeded.")
                raise

            wait_time = delay  # Default to the current exponential backoff delay.

            if is_rate_limit_error:
                retry_after_header = e.response.headers.get('Retry-After')
                if retry_after_header:
                    try:
                        # If the server provides a specific wait time, use it.
                        wait_time = int(retry_after_header)
                        logger.info(f"Rate limit hit. Honoring 'Retry-After' header, waiting for {wait_time} seconds.")
                    except (ValueError, TypeError):
                        logger.warning(f"Could not parse 'Retry-After' header: '{retry_after_header}'. Using exponential backoff.")
            
            jitter = random.uniform(0, 1)
            total_wait = wait_time + jitter
            
            logger.info(f"Retrying in {total_wait:.2f} seconds.")
            await asyncio.sleep(total_wait)

            delay *= exponential_backoff_factor

        except (ValueError, Exception) as e:
            logger.error(f"Non-retryable application error: {e}")
            raise

    raise Exception("API call failed after max retries.")