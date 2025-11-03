# utils/gemini_client.py

import os
import time
import google.generativeai as genai
from google.api_core import exceptions

# -------------------------------------------------------------------
# Gemini API setup
# -------------------------------------------------------------------
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError("❌ GEMINI_API_KEY not found in environment variables.")
genai.configure(api_key=api_key)

# Use your default model (you can adjust if you’re using 2.0 / 2.5 Pro)
MODEL_NAME = "gemini-2.5-pro"

# -------------------------------------------------------------------
# Basic query function
# -------------------------------------------------------------------
def query_gemini(prompt: str, temperature: float = 0.3) -> str:
    """
    Basic Gemini API call (no quota protection).
    """
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt, generation_config={"temperature": temperature})
    return response.text.strip() if response and response.text else ""

# -------------------------------------------------------------------
# Quota-aware / retry-safe Gemini query
# -------------------------------------------------------------------
def query_gemini_quota_safe(prompt: str, temperature: float = 0.3, retries: int = 3, delay: int = 10) -> str:
    """
    Quota-aware Gemini query with retry logic for transient errors or quota exhaustion.
    Automatically retries on ResourceExhausted, DeadlineExceeded, or other API errors.
    """
    for attempt in range(retries):
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            response = model.generate_content(prompt, generation_config={"temperature": temperature})
            if response and response.text:
                return response.text.strip()
            else:
                return "[Error] Gemini returned empty response."
        except exceptions.ResourceExhausted:
            print(f"[Gemini] ⚠️ Quota exhausted. Waiting {delay}s before retry {attempt + 1}/{retries}...")
            time.sleep(delay)
        except exceptions.DeadlineExceeded:
            print(f"[Gemini] ⏳ Timeout. Retrying ({attempt + 1}/{retries})...")
            time.sleep(delay)
        except Exception as e:
            print(f"[Gemini] Unexpected error: {e}. Retrying ({attempt + 1}/{retries})...")
            time.sleep(delay)
    return "[Error] Gemini quota exhausted after retries."

query_gemini_with_retry = query_gemini_quota_safe
