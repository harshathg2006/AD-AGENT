import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

class Config:
    GEMINI_API_KEY =  os.getenv("GEMINI_API_KEY")
