#!/usr/bin/env python3
"""
UWCI Sermon AI - Main Application with Enhanced Features
"""

import sys
import os
from pathlib import Path
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

# Import enhanced utilities
from app.utils.error_handler import handle_errors, ErrorHandler
from app.utils.logger import logger

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

# Enhanced Session Manager
class EnhancedSessionManager:
    """Enhanced session management with automatic cleanup and monitoring"""
    
    def __init__(self, sessions_file: str = "app/data/sessions.json"):
        self.sessions_file = Path(sessions_file)
        self.backup_file = Path(sessions_file.replace('.json', '_backup.json'))
        self.session_timeout = 24 * 60 * 60  # 24 hours
        self.max_sessions_per_user = 5  # Limit concurrent sessions
        
        # Ensure directory exists
        self.sessions_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initial cleanup
        self._cleanup_expired_sessions()
    
    def _load_sessions(self) -> dict:
        """Safely load sessions with backup recovery"""
        # Try main file first
        if self.sessions_file.exists():
            try:
                with open(self.sessions_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        sessions = json.loads(content)
                        logger.log_app_event("sessions_loaded", {"count": len(sessions)})
                        return sessions
            except (json.JSONDecodeError, IOError) as e:
                logger.log_app_event("sessions_load_failed", {"error": str(e)}, level="warning")
        
        # Try backup file
        if self.backup_file.exists():
            try:
                with open(self.backup_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        sessions = json.loads(content)
                        self._save_sessions(sessions)  # Restore main file
                        logger.log_app_event("sessions_recovered_from_backup", {"count": len(sessions)})
                        return sessions
            except (json.JSONDecodeError, IOError) as e:
                logger.log_app_event("backup_sessions_load_failed", {"error": str(e)}, level="error")
        
        logger.log_app_event("creating_new_sessions_file")
        return {}
    
    def _save_sessions(self, sessions: dict):
        """Safely save sessions with atomic write and backup"""
        try:
            # Create backup first
            if self.sessions_file.exists():
                import shutil
                shutil.copy2(self.sessions_file, self.backup_file)
            
            # Atomic write using temporary file
            temp_file = self.sessions_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(sessions, f, indent=2)
            
            # Move temp file to actual file (atomic operation)
            temp_file.replace(self.sessions_file)
            
            logger.log_app_event("sessions_saved", {"count": len(sessions)})
            
        except Exception as e:
            logger.log_app_event("sessions_save_failed", {"error": str(e)}, level="error")
    
    def _cleanup_expired_sessions(self):
        """Remove expired sessions and limit per-user sessions"""
        try:
            sessions = self._load_sessions()
            current_time = time.time()
            
            # Track sessions per user
            user_sessions = {}
            
            # Filter out expired sessions and track by user
            active_sessions = {}
            for session_id, session_data in sessions.items():
                if current_time - session_data.get('created_at', 0) < self.session_timeout:
                    username = session_data.get('username', '')
                    if username not in user_sessions:
                        user_sessions[username] = []
                    user_sessions[username].append((session_id, session_data))
            
            # Limit sessions per user (keep most recent)
            for username, user_session_list in user_sessions.items():
                # Sort by last accessed time (most recent first)
                user_session_list.sort(key=lambda x: x[1].get('last_accessed', 0), reverse=True)
                
                # Keep only the most recent sessions
                for session_id, session_data in user_session_list[:self.max_sessions_per_user]:
                    active_sessions[session_id] = session_data
            
            # Save cleaned sessions
            if len(active_sessions) != len(sessions):
                self._save_sessions(active_sessions)
                cleaned_count = len(sessions) - len(active_sessions)
                logger.log_app_event("sessions_cleaned", {"removed": cleaned_count})
                
        except Exception as e:
            logger.log_app_event("session_cleanup_failed", {"error": str(e)}, level="error")
    
    def create_session(self, username: str, user_agent: str = "", ip_address: str = "") -> str:
        """Create a new session with enhanced metadata"""
        import hashlib
        import secrets
        
        sessions = self._load_sessions()
        
        # Generate unique session ID
        session_data = f"{username}_{time.time()}_{secrets.token_hex(16)}"
        session_id = hashlib.sha256(session_data.encode()).hexdigest()
        
        # Check if user has too many sessions
        username_lower = username.lower()
        user_session_count = sum(1 for s in sessions.values() 
                               if s.get('username', '').lower() == username_lower)
        
        if user_session_count >= self.max_sessions_per_user:
            # Remove oldest session for this user
            user_sessions = [(sid, sdata) for sid, sdata in sessions.items() 
                           if sdata.get('username', '').lower() == username_lower]
            if user_sessions:
                oldest_session = min(user_sessions, key=lambda x: x[1].get('last_accessed', 0))
                del sessions[oldest_session[0]]
                logger.log_app_event("old_session_removed", {"username": username})
        
        # Add new session
        sessions[session_id] = {
            'username': username,
            'created_at': time.time(),
            'last_accessed': time.time(),
            'user_agent': user_agent,
            'ip_address': ip_address,
            'activity_count': 0
        }
        
        self._save_sessions(sessions)
        
        logger.log_auth_event("session_created", username, True, {
            "session_id": session_id[:8],
            "ip_address": ip_address
        })
        
        return session_id
    
    def validate_session(self, session_id: str, user_agent: str = "") -> str:
        """Validate session and update activity"""
        if not session_id:
            return None
        
        sessions = self._load_sessions()
        
        if session_id not in sessions:
            return None
        
        session_data = sessions[session_id]
        current_time = time.time()
        
        # Check if session has expired
        if current_time - session_data.get('created_at', 0) > self.session_timeout:
            self.destroy_session(session_id)
            logger.log_auth_event("session_expired", session_data.get('username', 'unknown'), False)
            return None
        
        # Update session activity
        session_data['last_accessed'] = current_time
        session_data['activity_count'] = session_data.get('activity_count', 0) + 1
        
        sessions[session_id] = session_data
        self._save_sessions(sessions)
        
        return session_data.get('username')
    
    def destroy_session(self, session_id: str):
        """Remove a specific session"""
        if not session_id:
            return
        
        sessions = self._load_sessions()
        
        if session_id in sessions:
            username = sessions[session_id].get('username', 'unknown')
            del sessions[session_id]
            self._save_sessions(sessions)
            logger.log_auth_event("session_destroyed", username, True)

# Global session manager
session_manager = EnhancedSessionManager()
error_handler = ErrorHandler()

@handle_errors(context="persistent_login", user_message="Login session error")
def check_persistent_login():
    """Check if user has a valid persistent session"""
    session_id = st.query_params.get('session', None)
    
    if session_id:
        user_agent = ""
        try:
            if hasattr(st, 'context') and hasattr(st.context, 'headers'):
                user_agent = st.context.headers.get("User-Agent", "")
        except:
            pass
        
        username = session_manager.validate_session(session_id, user_agent)
        if username:
            st.session_state.authenticated = True
            st.session_state.username = user_manager.get_display_name(username)
            st.session_state.user_role = user_manager.get_user_role(username)
            st.session_state.session_id = session_id
            
            # Log successful session restoration
            logger.log_auth_event("session_restored", username, True)
            logger.log_user_activity("session_start", {'session_id': session_id[:8]})
            
            return True
    
    return False

@handle_errors(context="create_session", user_message="Failed to create login session")
def create_persistent_login(username: str):
    """Create a persistent login session"""
    user_agent = ""
    ip_address = "unknown"
    
    try:
        if hasattr(st, 'context') and hasattr(st.context, 'headers'):
            user_agent = st.context.headers.get("User-Agent", "")
            ip_address = st.context.headers.get("X-Forwarded-For", "unknown")
    except:
        pass
    
    session_id = session_manager.create_session(username.lower(), user_agent, ip_address)
    st.session_state.session_id = session_id
    st.query_params['session'] = session_id

@handle_errors(context="logout", user_message="Logout failed")
def logout_persistent():
    """Logout and clear persistent session"""
    # Destroy server-side session
    if hasattr(st.session_state, 'session_id'):
        session_manager.destroy_session(st.session_state.session_id)
    
    # Clear query parameters
    st.query_params.clear()
    
    # Log logout
    if hasattr(st.session_state, 'username'):
        logger.log_auth_event("logout", st.session_state.username, True)
        logger.log_user_activity("logout")
    
    # Clear session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    st.rerun()

@handle_errors(context="main_app", user_message="Application error")
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
            try:
                st.session_state.ai_engine = AIEngine()
                st.session_state.sermon_processor = SermonProcessor()
                # Set API key from config file
                st.session_state.grok_api_key = GROK_API_KEY
                st.session_state.use_grok = True
                
                logger.log_app_event("ai_engine_initialized", {
                    "user": st.session_state.username,
                    "has_grok_key": bool(GROK_API_KEY)
                })
            except Exception as e:
                logger.log_app_event("ai_engine_init_failed", {"error": str(e)}, level="error")
                st.error("Failed to initialize AI engine. Please refresh the page.")
                st.stop()
    
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
        logger.log_user_activity("tab_accessed", {"tab": "ask_questions"})
        show_chat_tab()
    
    with tabs[1]:
        # Sermon Library tab
        logger.log_user_activity("tab_accessed", {"tab": "library"})
        show_library_tab()
    
    with tabs[2]:
        # Settings tab
        logger.log_user_activity("tab_accessed", {"tab": "settings"})
        show_settings_tab()
    
    # Admin tab (only if user is administrator)
    if st.session_state.get('user_role') == 'administrator' and len(tabs) > 3:
        with tabs[3]:
            logger.log_user_activity("tab_accessed", {"tab": "admin_settings"})
            from app.ui.admin_settings import show_admin_settings_tab
            show_admin_settings_tab()

@handle_errors(context="main_application", user_message="Application startup failed")
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
            
    except Exception as e:
        logger.log_app_event("icon_load_failed", {"error": str(e)}, level="warning")
    
    st.set_page_config(
        page_title="UWCI Sermon AI",
        page_icon=page_icon,
        layout="wide"
    )
    
    # Log application start
    logger.log_app_event("application_started")
    
    # Initialize authentication
    init_auth()
    
    # Check for persistent login first
    if not st.session_state.authenticated:
        if check_persistent_login():
            st.rerun()
    
    # Show login or main app based on authentication
    if not st.session_state.authenticated:
        logger.log_user_activity("login_page_shown")
        show_login_page()
    else:
        logger.log_user_activity("main_app_accessed", {
            "user": st.session_state.username,
            "role": st.session_state.user_role
        })
        main_app()

if __name__ == "__main__":
    main()