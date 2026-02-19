"""
Configuration Module

Manages environment variables and configuration settings for the application.
"""

import os
from pathlib import Path
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv()


class Config:
    """Application configuration."""
    
    NEO4J_URI = os.getenv("NEO4J_URI", "")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
    
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    MAX_RECORDS = 50
    TOP_K_RESULTS = 5
    
    DATA_DIR = BASE_DIR / "data"
    EMBEDDINGS_DIR = BASE_DIR / "embeddings"
    
    @classmethod
    def validate(cls):
        """Validate required configuration."""
        required = [
            cls.NEO4J_URI,
            cls.NEO4J_USERNAME, 
            cls.NEO4J_PASSWORD,
            cls.OPENROUTER_API_KEY
        ]
        return all(required)


config = Config()
