# utils.py

import logging
import time
import random
import json
import re
import base64
import asyncio
from typing import Optional

import httpx
from dateutil import parser

import config
from schemas import APIRequestBody, APIRequestMessage, APIRequestMessageContent, APIResponseBody

logger = logging.getLogger(__name__)
number_of_api_retries = config.API_RETRIES

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
    """Configures application-wide logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logger.info("Logging configured.")


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
    Calls the external LLM API asynchronously with exponential backoff.

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
            
            if is_server_error or isinstance(e, httpx.RequestError):
                logger.warning(f"API call failed (Attempt {i + 1}/{max_retries}): {e}. Retrying in {delay:.2f}s...")
                if i == max_retries - 1:
                    raise 
                await asyncio.sleep(delay)
                delay *= 2
            else:
                logger.error(f"Non-retryable client error: {e}")
                raise
        except (ValueError, Exception) as e:
            logger.error(f"Non-retryable application error: {e}")
            raise
            
    raise Exception("API call failed after max retries.")