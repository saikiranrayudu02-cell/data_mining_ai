import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings

# Resolve paths relative to app
BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "DataMine AI Classifier API"
    API_V1_STR: str = "/api/v1"
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    
    # Storage settings
    UPLOAD_DIR: Path = BASE_DIR / "storage" / "uploads"
    MODEL_DIR: Path = BASE_DIR / "storage" / "models"
    PLOT_DIR: Path = BASE_DIR / "storage" / "plots"
    EXPORT_DIR: Path = BASE_DIR / "storage" / "exports"
    
    # ML settings
    MAX_FILE_SIZE_MB: int = 50
    
    class Config:
        case_sensitive = True

# Instantiate settings
settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.MODEL_DIR, exist_ok=True)
os.makedirs(settings.PLOT_DIR, exist_ok=True)
os.makedirs(settings.EXPORT_DIR, exist_ok=True)
