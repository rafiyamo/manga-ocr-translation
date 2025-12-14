"""
Configuration utilities (e.g., loading API keys).
"""

from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")