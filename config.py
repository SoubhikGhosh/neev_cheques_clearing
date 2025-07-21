# config.py

# --- API Configuration ---
API_URL = 'https://10.216.70.62/DEV/litellm/chat/completions'
API_KEY = 'abcd'
MODEL_NAME = "gemini-2.5-flash"

# --- Performance Configuration ---
# Limits the number of concurrent API requests to the LLM to prevent rate-limiting.
API_CONCURRENCY_LIMIT = 50

# Defines the output directory for the final reports.
OUTPUT_DIR = "job_outputs"

# --- Retry Configuration ---
API_RETRIES = 50
JSON_RETRIES = 3

# --- Field Definitions ---
FIELDS = [
    {"id": 1, "name": "bank_name"},
    {"id": 2, "name": "bank_branch"},
    {"id": 3, "name": "account_number"},
    {"id": 4, "name": "date"},
    {"id": 5, "name": "payee_name"},
    {"id": 6, "name": "amount_words"},
    {"id": 7, "name": "amount_numeric"},
    {"id": 8, "name": "currency"},
    {"id": 9, "name": "issuer_name"},
    {"id": 10, "name": "IFSC"},
    {"id": 11, "name": "micr_scan_instrument_number"},
    {"id": 12, "name": "micr_scan_payee_details"},
    {"id": 13, "name": "micr_scan_micr_acno"},
    {"id": 14, "name": "micr_scan_instrument_type"}
]