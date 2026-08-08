import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings

# Resolve paths relative to app
BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "DataMine AI Classifier API"
    API_V1_STR: str = "/api/v1"

    # Server settings – Render injects $PORT; fall back to 8000 for local dev
    PORT: int = int(os.environ.get("PORT", 8000))

    # CORS settings – override via ALLOWED_ORIGINS env var on Render (comma-separated)
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://data-mining-ai-backend.onrender.com",
        "*",  # Allow all during initial deployment; tighten once frontend URL is known
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
