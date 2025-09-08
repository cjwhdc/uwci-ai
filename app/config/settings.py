import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SERMONS_DIR = DATA_DIR / "sermons" 
BIBLE_DIR = DATA_DIR / "bible"
DB_DIR = PROJECT_ROOT / "sermon_db"
LOGS_DIR = PROJECT_ROOT / "logs"

# AI Settings
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
MAX_SEARCH_RESULTS = 10

# Supported file formats
SUPPORTED_FORMATS = ['.md', '.txt']

# Ollama settings
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama2"

# Create directories if they don't exist
for directory in [DATA_DIR, SERMONS_DIR, BIBLE_DIR, DB_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)