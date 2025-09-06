#!/usr/bin/env python3
"""
Setup script for UWCI Sermon AI
Run this after cloning from GitHub to set up the application
"""

import os
import shutil
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    directories = [
        "app/data",
        "data/sermons", 
        "logs",
        "backup"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def copy_config_template():
    """Copy configuration template from setup folder if config doesn't exist"""
    template_path = "setup/config.template.py"
    config_path = "app/config/config.py"
    
    if not os.path.exists(config_path):
        if os.path.exists(template_path):
            shutil.copy2(template_path, config_path)
            print(f"Created {config_path} from setup template")
            print("⚠️  IMPORTANT: Edit app/config/config.py with your actual API keys!")
        else:
            print(f"Warning: Template file {template_path} not found")
    else:
        print(f"Configuration file {config_path} already exists")

def copy_env_template():
    """Copy .env template from setup folder if .env doesn't exist"""
    template_path = "setup/.env.example"
    env_path = ".env"
    
    if not os.path.exists(env_path):
        if os.path.exists(template_path):
            shutil.copy2(template_path, env_path)
            print(f"Created {env_path} from setup template")
            print("⚠️  IMPORTANT: Edit .env with your actual API keys!")
        else:
            print(f"Warning: Template file {template_path} not found")
    else:
        print(f"Environment file {env_path} already exists")

def create_init_files():
    """Create __init__.py files for Python packages"""
    init_files = [
        "app/__init__.py",
        "app/ui/__init__.py", 
        "app/config/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"Created {init_file}")

def set_permissions():
    """Set appropriate file permissions (Unix/Linux/Mac only)"""
    try:
        # Secure data directory
        os.chmod("app/data", 0o700)
        
        # Secure config file if it exists
        if os.path.exists("app/config/config.py"):
            os.chmod("app/config/config.py", 0o600)
            
        # Secure .env file if it exists  
        if os.path.exists(".env"):
            os.chmod(".env", 0o600)
            
        print("Set secure file permissions")
    except (OSError, AttributeError):
        print("Could not set file permissions (Windows or permission error)")

def check_gitignore():
    """Check if .gitignore exists and has required entries"""
    gitignore_path = ".gitignore"
    required_entries = [
        "app/config/config.py",
        "app/data/",
        ".env"
    ]
    
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as f:
            content = f.read()
            
        missing = [entry for entry in required_entries if entry not in content]
        if missing:
            print(f"⚠️  Warning: .gitignore may be missing these entries:")
            for entry in missing:
                print(f"   {entry}")
    else:
        print("⚠️  Warning: No .gitignore file found. Create one to protect sensitive files!")

def main():
    """Main setup function"""
    print("Setting up UWCI Sermon AI...")
    print("=" * 50)
    
    create_directories()
    create_init_files()
    copy_config_template()
    copy_env_template()
    set_permissions()
    check_gitignore()
    
    print("\n" + "=" * 50)
    print("Setup complete!")
    print("\nNext steps:")
    print("1. Edit app/config/config.py with your actual Grok API key")
    print("2. Or edit .env file with environment variables")
    print("3. Add your sermon files to data/sermons/")
    print("4. Run: streamlit run main.py")
    print("\n⚠️  Remember: Never commit config.py or .env to version control!")
    print("\nTemplates are available in the setup/ folder for reference.")

if __name__ == "__main__":
    main()