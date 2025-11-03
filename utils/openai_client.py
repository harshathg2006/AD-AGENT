# utils/openai_client.py

import google.generativeai as genai
from config.config import Config

genai.configure(api_key=Config.GEMINI_API_KEY)

def query_openai(messages, model="gemini-pro"):
    prompt = "\n".join([msg["content"] for msg in messages if msg["role"] == "user"])
    response = genai.GenerativeModel(model).generate_content(prompt)
    return response.text
