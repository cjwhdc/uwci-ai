#!/usr/bin/env python3
"""
UWCI Sermon AI - Main Application
"""

import sys
import os
from pathlib import Path
import hashlib
import time
import json

# Suppress HuggingFace tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add the project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
from app.auth import init_auth, logout, user_manager
from app.ui.login import show_login_page
from app.ui.chat import show_chat_tab
from app.ui.library import show_library_tab
from app.ui.settings import show_settings_tab
from app.ai_engine import AIEngine
from app.sermon_processor import SermonProcessor

# Import configuration
try:
    from app.config.config import GROK_API_KEY
except ImportError:
    # Config file doesn't exist (likely on Streamlit Cloud)
    # Check for Streamlit secrets or environment variables
    import os
    GROK_API_KEY = None
    
    # Try Streamlit secrets first (safely)
    try:
        if hasattr(st, 'secrets') and 'GROK_API_KEY' in st.secrets:
            GROK_API_KEY = st.secrets['GROK_API_KEY']
    except Exception:
        pass
    
    # Fall back to environment variable
    if not GROK_API_KEY and 'GROK_API_KEY' in os.environ:
        GROK_API_KEY = os.environ['GROK_API_KEY']
    
    if not GROK_API_KEY:
        st.error("GROK_API_KEY not found. Please set it in Streamlit Cloud secrets or as an environment variable.")
        st.stop()

# Core dependencies check
try:
    import chromadb
    import openai
    from sentence_transformers import SentenceTransformer
    import requests
except ImportError as e:
    st.error(f"Missing required packages. Please install: {e}")
    st.stop()

class SessionManager:
    """Manage persistent login sessions"""
    
    def __init__(self):
        self.sessions_file = Path("app/data/sessions.json")
        self.session_timeout = 24 * 60 * 60  # 24 hours in seconds
        self._cleanup_expired_sessions()
    
    def _cleanup_expired_sessions(self):
        """Remove expired sessions"""
        if not self.sessions_file.exists():
            return
        
        try:
            with open(self.sessions_file, 'r') as f:
                sessions = json.load(f)
            
            current_time = time.time()
            active_sessions = {
                session_id: session_data 
                for session_id, session_data in sessions.items()
                if current_time - session_data.get('created_at', 0) < self.session_timeout
            }
            
            with open(self.sessions_file, 'w') as f:
                json.dump(active_sessions, f, indent=2)
        except:
            # If there's any error reading/writing sessions, start fresh
            if self.sessions_file.exists():
                self.sessions_file.unlink()
    
    def create_session(self, username: str) -> str:
        """Create a new session for the user"""
        # Create a unique session ID based on username and timestamp
        session_data = f"{username}_{time.time()}_{os.urandom(16).hex()}"
        session_id = hashlib.sha256(session_data.encode()).hexdigest()
        
        # Load existing sessions
        sessions = {}
        if self.sessions_file.exists():
            try:
                with open(self.sessions_file, 'r') as f:
                    sessions = json.load(f)
            except:
                sessions = {}
        
        # Add new session
        sessions[session_id] = {
            'username': username,
            'created_at': time.time(),
            'last_accessed': time.time()
        }
        
        # Save sessions
        self.sessions_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.sessions_file, 'w') as f:
            json.dump(sessions, f, indent=2)
        
        return session_id
    
    def validate_session(self, session_id: str) -> str:
        """Validate session and return username if valid"""
        if not session_id or not self.sessions_file.exists():
            return None
        
        try:
            with open(self.sessions_file, 'r') as f:
                sessions = json.load(f)
            
            if session_id not in sessions:
                return None
            
            session_data = sessions[session_id]
            current_time = time.time()
            
            # Check if session has expired
            if current_time - session_data.get('created_at', 0) > self.session_timeout:
                self.destroy_session(session_id)
                return None
            
            # Update last accessed time
            session_data['last_accessed'] = current_time
            with open(self.sessions_file, 'w') as f:
                json.dump(sessions, f, indent=2)
            
            return session_data.get('username')
        except:
            return None
    
    def destroy_session(self, session_id: str):
        """Remove a session"""
        if not session_id or not self.sessions_file.exists():
            return
        
        try:
            with open(self.sessions_file, 'r') as f:
                sessions = json.load(f)
            
            if session_id in sessions:
                del sessions[session_id]
                
                with open(self.sessions_file, 'w') as f:
                    json.dump(sessions, f, indent=2)
        except:
            pass

# Global session manager
session_manager = SessionManager()

def check_persistent_login():
    """Check if user has a valid persistent session"""
    # Get session ID from query params (Streamlit's way of persisting data)
    session_id = st.query_params.get('session', None)
    
    if session_id:
        username = session_manager.validate_session(session_id)
        if username:
            # Restore session state
            st.session_state.authenticated = True
            st.session_state.username = user_manager.get_display_name(username)
            st.session_state.user_role = user_manager.get_user_role(username)
            st.session_state.session_id = session_id
            return True
    
    return False

def create_persistent_login(username: str):
    """Create a persistent login session"""
    session_id = session_manager.create_session(username.lower())
    st.session_state.session_id = session_id
    
    # Set query parameter to persist session across page refreshes
    st.query_params['session'] = session_id

def logout_persistent():
    """Logout and clear persistent session"""
    # Destroy server-side session
    if hasattr(st.session_state, 'session_id'):
        session_manager.destroy_session(st.session_state.session_id)
    
    # Clear query parameters
    st.query_params.clear()
    
    # Clear session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    st.rerun()

def main_app():
    """Main application after authentication"""
    # Display title with custom icon
    col1, col2 = st.columns([1, 10])
    
    with col1:
        try:
            from PIL import Image
            icon_path = "assets/icon.jpg"
            if Path(icon_path).exists():
                icon = Image.open(icon_path)
                # Resize for display in the UI (larger than favicon)
                if icon.mode != 'RGB':
                    icon = icon.convert('RGB')
                # Make it a good size for the title area
                icon = icon.resize((48, 48), Image.Resampling.LANCZOS)
                st.image(icon, width=48)
            else:
                st.markdown("📖")
        except Exception:
            st.markdown("📖")
    
    with col2:
        st.title("UWCI Sermon AI")
    
    # Initialize components
    if 'ai_engine' not in st.session_state:
        with st.spinner("Initializing AI engine..."):
            st.session_state.ai_engine = AIEngine()
            st.session_state.sermon_processor = SermonProcessor()
            # Set API key from config file
            st.session_state.grok_api_key = GROK_API_KEY
            st.session_state.use_grok = True
    
    # Show modern sidebar
    from app.ui.sidebar import show_sidebar
    show_sidebar()
    
    # Main interface tabs - conditional based on user role
    if st.session_state.get('user_role') == 'administrator':
        tabs = st.tabs(["Ask Questions", "Sermon Library", "Settings", "Admin Settings"])
    else:
        tabs = st.tabs(["Ask Questions", "Sermon Library", "Settings"])
    
    # Only render content in the active tab to prevent duplicate elements
    with tabs[0]:
        # Ask Questions tab
        show_chat_tab()
    
    with tabs[1]:
        # Sermon Library tab
        show_library_tab()
    
    with tabs[2]:
        # Settings tab
        show_settings_tab()
    
    # Admin tab (only if user is administrator)
    if st.session_state.get('user_role') == 'administrator' and len(tabs) > 3:
        with tabs[3]:
            from app.ui.admin_settings import show_admin_settings_tab
            show_admin_settings_tab()

def main():
    """Main application entry point"""
    # Load custom icon
    page_icon = "📖"  # Default fallback
    
    try:
        from PIL import Image
        icon_path = "assets/icon.jpg"
        if Path(icon_path).exists():
            img = Image.open(icon_path)
            
            # Ensure RGB mode and proper size for favicon
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            if img.size[0] > 64 or img.size[1] > 64:
                img = img.resize((64, 64), Image.Resampling.LANCZOS)
            
            page_icon = img
            
    except Exception:
        # If loading fails, stick with emoji fallback
        pass
    
    st.set_page_config(
        page_title="UWCI Sermon AI",
        page_icon=page_icon,
        layout="wide"
    )
    
    # Initialize authentication
    init_auth()
    
    # Check for persistent login first
    if not st.session_state.authenticated:
        if check_persistent_login():
            st.rerun()
    
    # Show login or main app based on authentication
    if not st.session_state.authenticated:
        show_login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()