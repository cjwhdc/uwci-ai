# setup_enhanced_features.py
"""
Setup script to create directories and initialize enhanced features
Run this after adding the enhanced utility files
"""

import os
from pathlib import Path

def create_directories():
    """Create required directories for enhanced features"""
    directories = [
        "app/utils",
        "logs",
        "app/data",
        "data/sermons",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create __init__.py files for Python packages
    init_files = [
        "app/utils/__init__.py",
        "tests/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"Created file: {init_file}")

def create_env_example():
    """Create example environment file"""
    env_content = """# Environment variables for Sermon AI Enhanced Features
# Copy this to .env and fill in your actual values

# API Keys
GROK_API_KEY=your-grok-api-key-here

# Enhanced Features (true/false)
ENABLE_ENHANCED_LOGGING=true
ENABLE_RATE_LIMITING=true
ENABLE_SESSION_MONITORING=true

# Rate Limiting Settings
GLOBAL_REQUESTS_PER_MINUTE=20
USER_REQUESTS_PER_MINUTE=5
GLOBAL_REQUESTS_PER_HOUR=1000
USER_REQUESTS_PER_HOUR=50

# Session Management
MAX_SESSIONS_PER_USER=5
SESSION_CLEANUP_INTERVAL=3600
SESSION_TIMEOUT=86400

# Logging Configuration
LOG_RETENTION_DAYS=30
LOG_LEVEL=INFO

# Performance Settings
QUERY_CACHE_TIMEOUT=300
MAX_CACHE_SIZE=100

# Dropbox Sync (optional)
USE_DROPBOX_SYNC=false
DROPBOX_ACCESS_TOKEN=your-dropbox-token-here

# Database Configuration
DATABASE_PATH=./sermon_db
USER_DATA_PATH=app/data/users.json

# Security Settings
MAX_LOGIN_ATTEMPTS=5
LOCKOUT_DURATION=900
"""
    
    with open(".env.example", "w") as f:
        f.write(env_content)
    print("Created .env.example file")

def create_test_files():
    """Create basic test files"""
    
    # Test for error handling
    test_error_content = '''# tests/test_error_handling.py
import pytest
import sys
sys.path.append('..')

from app.utils.error_handler import ErrorHandler, handle_errors

def test_error_handler_basic():
    """Test basic error handling functionality"""
    handler = ErrorHandler()
    
    try:
        raise ValueError("Test error")
    except Exception as e:
        error_id = handler.handle_exception(e, "test_context")
        assert len(error_id) > 0
        print(f"Error handled with ID: {error_id}")

def test_error_decorator():
    """Test error handling decorator"""
    @handle_errors(context="test_function")
    def failing_function():
        raise RuntimeError("Decorator test")
    
    # Should not raise exception due to decorator
    result = failing_function()
    assert result is None
    print("Decorator test passed")

if __name__ == "__main__":
    test_error_handler_basic()
    test_error_decorator()
    print("All error handling tests passed!")
'''
    
    with open("tests/test_error_handling.py", "w") as f:
        f.write(test_error_content)
    print("Created tests/test_error_handling.py")
    
    # Test for logging
    test_logging_content = '''# tests/test_logging.py
import sys
sys.path.append('..')

from app.utils.logger import logger, log_function_call

def test_basic_logging():
    """Test basic logging functionality"""
    logger.log_app_event("test_event", {"test": "data"})
    logger.log_auth_event("test_auth", "test_user", True)
    logger.log_user_activity("test_activity", {"action": "test"})
    print("Basic logging test passed")

@log_function_call("test")
def test_logged_function():
    """Test function with logging decorator"""
    return "test_result"

if __name__ == "__main__":
    test_basic_logging()
    result = test_logged_function()
    print(f"Logged function returned: {result}")
    print("All logging tests passed!")
'''
    
    with open("tests/test_logging.py", "w") as f:
        f.write(test_logging_content)
    print("Created tests/test_logging.py")

def create_gitignore_additions():
    """Create or update .gitignore with new entries"""
    gitignore_additions = """
# Enhanced Features - Logs and Data
logs/
*.log
app/data/sessions.json
app/data/rate_limits.json
app/data/users_backup.json

# Environment files
.env

# Cache files
__pycache__/
*.pyc
*.pyo

# Test artifacts
.pytest_cache/
"""
    
    gitignore_path = Path(".gitignore")
    
    if gitignore_path.exists():
        # Read existing content
        with open(gitignore_path, "r") as f:
            existing_content = f.read()
        
        # Check if our additions are already there
        if "Enhanced Features" not in existing_content:
            with open(gitignore_path, "a") as f:
                f.write(gitignore_additions)
            print("Updated existing .gitignore file")
        else:
            print(".gitignore already contains enhanced features entries")
    else:
        with open(gitignore_path, "w") as f:
            f.write(gitignore_additions.strip())
        print("Created new .gitignore file")

def create_readme_update():
    """Create README update with enhanced features info"""
    readme_update = """
# Enhanced Features Added

This Sermon AI application now includes enhanced features for production use:

## New Features

### 1. Enhanced Error Handling
- User-friendly error messages
- Comprehensive error logging with unique IDs
- Automatic error context capture
- Graceful degradation on failures

### 2. Comprehensive Logging
- Separate logs for different system components
- User activity tracking
- Authentication event logging
- Performance metrics logging
- Automatic log rotation and cleanup

### 3. Enhanced Session Management
- Automatic session cleanup
- Per-user session limits
- Session activity tracking
- Better security with user agent validation

### 4. Performance Improvements
- Query result caching
- Background cleanup tasks
- Optimized database operations
- Memory usage monitoring

## Configuration

Copy `.env.example` to `.env` and configure your settings:

```bash
cp .env.example .env
# Edit .env with your actual values
```

## Log Files

The application creates structured log files in the `logs/` directory:
- `app_YYYYMMDD.log` - General application events
- `auth_YYYYMMDD.log` - Authentication and security events
- `database_YYYYMMDD.log` - Database operations
- `ai_YYYYMMDD.log` - AI model operations
- `activity_YYYYMMDD.log` - User activities
- `errors_YYYYMMDD.log` - Error details

## Testing

Run the enhanced feature tests:

```bash
python tests/test_error_handling.py
python tests/test_logging.py
```

## Monitoring

Administrators can monitor system health through:
- Log file analysis
- Session statistics in admin panel
- Error rate monitoring
- Performance metrics tracking

## Backwards Compatibility

All enhanced features are backwards compatible. Existing functionality remains unchanged while adding new capabilities.
"""
    
    with open("ENHANCED_FEATURES.md", "w") as f:
        f.write(readme_update.strip())
    print("Created ENHANCED_FEATURES.md documentation")

def check_existing_files():
    """Check which files need to be updated"""
    files_to_check = [
        "main.py",
        "app/ai_engine.py", 
        "app/auth.py",
        "app/config/config.py"
    ]
    
    print("Checking existing files that need updates:")
    for file_path in files_to_check:
        if Path(file_path).exists():
            print(f"  ✓ Found: {file_path} (needs enhancement)")
        else:
            print(f"  ✗ Missing: {file_path}")

def main():
    """Run the setup process"""
    print("Setting up enhanced features for Sermon AI...")
    print("=" * 60)
    
    print("1. Creating directories...")
    create_directories()
    print()
    
    print("2. Creating configuration examples...")
    create_env_example()
    print()
    
    print("3. Creating test files...")
    create_test_files()
    print()
    
    print("4. Updating .gitignore...")
    create_gitignore_additions()
    print()
    
    print("5. Creating documentation...")
    create_readme_update()
    print()
    
    print("6. Checking existing files...")
    check_existing_files()
    print()
    
    print("Setup complete!")
    print("=" * 60)
    print("Next steps:")
    print("1. Copy .env.example to .env and fill in your values")
    print("2. Create the utility files in app/utils/:")
    print("   - error_handler.py")
    print("   - logger.py")
    print("3. Update your existing files with enhanced versions:")
    print("   - main.py")
    print("   - app/ai_engine.py") 
    print("   - app/auth.py")
    print("   - app/config/config.py")
    print("4. Test the enhanced features:")
    print("   python tests/test_error_handling.py")
    print("   python tests/test_logging.py")
    print("5. Start your application:")
    print("   streamlit run main.py")
    print()
    print("See ENHANCED_FEATURES.md for detailed documentation.")

if __name__ == "__main__":
    main()