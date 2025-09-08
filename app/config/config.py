# Configuration for Sermon AI with Enhanced Features
# This file loads settings from environment variables (.env file)
# For Streamlit Cloud, set environment variables in the app dashboard

import os
from pathlib import Path

def load_env():
    """Load environment variables from .env file if it exists"""
    env_path = Path('.env')
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip('"\'')
                    os.environ[key] = value

# Load environment variables first
load_env()

# API Configuration
GROK_API_KEY = os.getenv('GROK_API_KEY')

# Streamlit Cloud uses secrets, so check there safely
try:
    import streamlit as st
    if hasattr(st, 'secrets') and 'GROK_API_KEY' in st.secrets:
        GROK_API_KEY = st.secrets['GROK_API_KEY']
except Exception:
    # Secrets not available (likely local development), continue with env vars
    pass

if not GROK_API_KEY:
    print("Warning: GROK_API_KEY not found. Set it in .env file or environment variables.")

# Enhanced features configuration
ENABLE_ENHANCED_LOGGING = os.getenv('ENABLE_ENHANCED_LOGGING', 'true').lower() == 'true'
ENABLE_RATE_LIMITING = os.getenv('ENABLE_RATE_LIMITING', 'true').lower() == 'true'
ENABLE_SESSION_MONITORING = os.getenv('ENABLE_SESSION_MONITORING', 'true').lower() == 'true'

# Rate limiting settings
GLOBAL_REQUESTS_PER_MINUTE = int(os.getenv('GLOBAL_REQUESTS_PER_MINUTE', '20'))
USER_REQUESTS_PER_MINUTE = int(os.getenv('USER_REQUESTS_PER_MINUTE', '5'))
GLOBAL_REQUESTS_PER_HOUR = int(os.getenv('GLOBAL_REQUESTS_PER_HOUR', '1000'))
USER_REQUESTS_PER_HOUR = int(os.getenv('USER_REQUESTS_PER_HOUR', '50'))

# Session management settings
MAX_SESSIONS_PER_USER = int(os.getenv('MAX_SESSIONS_PER_USER', '5'))
SESSION_CLEANUP_INTERVAL = int(os.getenv('SESSION_CLEANUP_INTERVAL', '3600'))  # 1 hour

# Logging settings
LOG_RETENTION_DAYS = int(os.getenv('LOG_RETENTION_DAYS', '30'))
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Dropbox sync configuration
USE_DROPBOX_SYNC = os.getenv('USE_DROPBOX_SYNC', 'false').lower() == 'true'
DROPBOX_ACCESS_TOKEN = os.getenv('DROPBOX_ACCESS_TOKEN')

# Check Streamlit secrets for Dropbox token safely
try:
    import streamlit as st
    if hasattr(st, 'secrets') and 'DROPBOX_ACCESS_TOKEN' in st.secrets:
        DROPBOX_ACCESS_TOKEN = st.secrets['DROPBOX_ACCESS_TOKEN']
        USE_DROPBOX_SYNC = True  # Enable if token found in secrets
except Exception:
    # Secrets not available, continue with env vars
    pass

class DropboxDatabaseSync:
    def __init__(self):
        self.local_db_path = Path("./sermon_db")
        self.dropbox_path = "/sermon_db_backup.zip"
        
    def sync_from_dropbox(self):
        """Download database from Dropbox"""
        if not USE_DROPBOX_SYNC or not DROPBOX_ACCESS_TOKEN:
            return False
            
        try:
            import dropbox
            import zipfile
            import shutil
            
            dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
            
            # Download the database zip file
            local_zip = "sermon_db_download.zip"
            
            try:
                metadata, response = dbx.files_download(self.dropbox_path)
                
                with open(local_zip, "wb") as f:
                    f.write(response.content)
                
                # Backup existing database if it exists
                if self.local_db_path.exists():
                    backup_path = f"{self.local_db_path}_backup"
                    if Path(backup_path).exists():
                        shutil.rmtree(backup_path)
                    shutil.move(str(self.local_db_path), backup_path)
                
                # Extract downloaded database
                with zipfile.ZipFile(local_zip, 'r') as zip_ref:
                    zip_ref.extractall("./")
                
                os.remove(local_zip)
                return True
                
            except dropbox.exceptions.ApiError as e:
                if e.error.is_path_not_found():
                    return False  # No backup exists yet
                else:
                    raise e
                    
        except Exception as e:
            print(f"Dropbox download failed: {e}")
            return False
    
    def sync_to_dropbox(self):
        """Upload database to Dropbox"""
        if not USE_DROPBOX_SYNC or not DROPBOX_ACCESS_TOKEN:
            return False
            
        if not self.local_db_path.exists():
            return False
            
        try:
            import dropbox
            import zipfile
            
            dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
            
            # Create a zip file of the database
            local_zip = "sermon_db_upload.zip"
            
            with zipfile.ZipFile(local_zip, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                for file_path in self.local_db_path.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(self.local_db_path.parent)
                        zip_ref.write(file_path, arcname)
            
            # Upload to Dropbox
            with open(local_zip, 'rb') as f:
                dbx.files_upload(
                    f.read(), 
                    self.dropbox_path, 
                    mode=dropbox.files.WriteMode('overwrite')
                )
            
            os.remove(local_zip)
            return True
            
        except Exception as e:
            print(f"Dropbox upload failed: {e}")
            return False
    
    def get_backup_info(self):
        """Get information about the Dropbox backup"""
        if not USE_DROPBOX_SYNC or not DROPBOX_ACCESS_TOKEN:
            return None
            
        try:
            import dropbox
            
            dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
            metadata = dbx.files_get_metadata(self.dropbox_path)
            
            return {
                'size': metadata.size,
                'modified': metadata.server_modified.strftime('%Y-%m-%d %H:%M:%S'),
                'exists': True
            }
        except dropbox.exceptions.ApiError as e:
            if e.error.is_path_not_found():
                return {'exists': False}
            return None
        except Exception:
            return None

# Text processing settings
DEFAULT_CHUNK_SIZE = int(os.getenv('DEFAULT_CHUNK_SIZE', '1000'))
DEFAULT_CHUNK_OVERLAP = int(os.getenv('DEFAULT_CHUNK_OVERLAP', '200'))

# Search settings
MAX_SEARCH_RESULTS = int(os.getenv('MAX_SEARCH_RESULTS', '10'))

# Database configuration
DATABASE_PATH = os.getenv('DATABASE_PATH', './sermon_db')
USER_DATA_PATH = os.getenv('USER_DATA_PATH', 'app/data/users.json')

# Security settings
SESSION_TIMEOUT = int(os.getenv('SESSION_TIMEOUT', '86400'))  # 24 hours
MAX_LOGIN_ATTEMPTS = int(os.getenv('MAX_LOGIN_ATTEMPTS', '5'))
LOCKOUT_DURATION = int(os.getenv('LOCKOUT_DURATION', '900'))  # 15 minutes

# Performance settings
QUERY_CACHE_TIMEOUT = int(os.getenv('QUERY_CACHE_TIMEOUT', '300'))  # 5 minutes
MAX_CACHE_SIZE = int(os.getenv('MAX_CACHE_SIZE', '100'))

# Debug print for troubleshooting
if __name__ == "__main__":
    print(f"GROK_API_KEY loaded: {bool(GROK_API_KEY)}")
    print(f"USE_DROPBOX_SYNC: {USE_DROPBOX_SYNC}")
    print(f"DROPBOX_ACCESS_TOKEN loaded: {bool(DROPBOX_ACCESS_TOKEN)}")
    print(f"Enhanced logging: {ENABLE_ENHANCED_LOGGING}")
    print(f"Rate limiting: {ENABLE_RATE_LIMITING}")
    print(f"Session monitoring: {ENABLE_SESSION_MONITORING}")