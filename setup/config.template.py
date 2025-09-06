# Configuration for Sermon AI
# This file loads settings from environment variables (.env file)
# Copy .env.example to .env and fill in your actual values

import os
from pathlib import Path

# Load environment variables from .env file
def load_env():
    """Load environment variables from .env file if it exists"""
    env_path = Path('.env')
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

# Load environment variables
load_env()

# API Configuration
GROK_API_KEY = os.getenv('GROK_API_KEY')

if not GROK_API_KEY or GROK_API_KEY == 'xai-your-actual-api-key-here':
    print("Warning: GROK_API_KEY not set. Please configure your .env file.")

# Text processing settings
DEFAULT_CHUNK_SIZE = int(os.getenv('DEFAULT_CHUNK_SIZE', '1000'))
DEFAULT_CHUNK_OVERLAP = int(os.getenv('DEFAULT_CHUNK_OVERLAP', '200'))

# Search settings
MAX_SEARCH_RESULTS = int(os.getenv('MAX_SEARCH_RESULTS', '5'))

# Database configuration
DATABASE_PATH = os.getenv('DATABASE_PATH', './sermon_db')
USER_DATA_PATH = os.getenv('USER_DATA_PATH', 'app/data/users.json')

# Security settings
SESSION_TIMEOUT = int(os.getenv('SESSION_TIMEOUT', '3600'))  # 1 hour
MAX_LOGIN_ATTEMPTS = int(os.getenv('MAX_LOGIN_ATTEMPTS', '5'))
LOCKOUT_DURATION = int(os.getenv('LOCKOUT_DURATION', '900'))  # 15 minutes

# Logging settings
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_DIR = os.getenv('LOG_DIR', 'logs')