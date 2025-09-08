# app/utils/error_handler.py
import logging
import streamlit as st
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import functools
import inspect

class ErrorHandler:
    """Centralized error handling with user-friendly messages and logging"""
    
    def __init__(self):
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create error-specific logger
        self.logger = logging.getLogger('sermon_ai_errors')
        self.logger.setLevel(logging.ERROR)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler for errors
        error_handler = logging.FileHandler(
            f'logs/errors_{datetime.now().strftime("%Y%m%d")}.log'
        )
        error_handler.setLevel(logging.ERROR)
        
        # Console handler for development
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        error_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(error_handler)
        self.logger.addHandler(console_handler)
    
    def handle_exception(self, exception: Exception, context: str = "", user_message: str = None) -> str:
        """Handle exceptions with logging and user-friendly messages"""
        error_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Get user context safely
        user_context = {}
        try:
            if hasattr(st.session_state, 'username'):
                user_context['username'] = st.session_state.username
            if hasattr(st.session_state, 'user_role'):
                user_context['user_role'] = st.session_state.user_role
        except:
            pass
        
        # Log the full error
        self.logger.error(f"Error ID {error_id} in {context}: {str(exception)}")
        self.logger.error(f"User context: {user_context}")
        self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Determine user-friendly message
        if user_message:
            display_message = user_message
        else:
            display_message = self._get_user_friendly_message(exception)
        
        # Show error to user with error ID for support
        st.error(f"{display_message} (Error ID: {error_id})")
        
        return error_id
    
    def _get_user_friendly_message(self, exception: Exception) -> str:
        """Convert technical exceptions to user-friendly messages"""
        error_msg = str(exception).lower()
        
        # Database errors
        if "readonly database" in error_msg or "permission denied" in error_msg:
            return "Database is currently read-only. Please try again or contact your administrator."
        
        if "missing metadata" in error_msg or "missing field" in error_msg:
            return "Database corruption detected. Contact your administrator to rebuild the database."
        
        # Network/API errors
        if "connection" in error_msg or "timeout" in error_msg:
            return "Network connection issue. Please check your internet connection and try again."
        
        if "401" in error_msg or "unauthorized" in error_msg:
            return "Authentication failed. Please check your API credentials."
        
        if "429" in error_msg or "rate limit" in error_msg:
            return "Rate limit exceeded. Please wait a moment before trying again."
        
        # File system errors
        if "file not found" in error_msg or "no such file" in error_msg:
            return "Required file not found. Please contact your administrator."
        
        if "insufficient space" in error_msg or "disk full" in error_msg:
            return "Insufficient disk space. Please contact your administrator."
        
        # Default message
        return "An unexpected error occurred. Please try again or contact support if the problem persists."

# Decorator for automatic error handling
def handle_errors(context: str = "", user_message: str = None):
    """Decorator to automatically handle errors in functions"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler = ErrorHandler()
                error_handler.handle_exception(e, context or func.__name__, user_message)
                return None
        return wrapper
    return decorator

# Specific error classes for better handling
class SermonProcessingError(Exception):
    """Raised when sermon processing fails"""
    pass

class DatabaseError(Exception):
    """Raised when database operations fail"""
    pass

class AIModelError(Exception):
    """Raised when AI model operations fail"""
    pass

class AuthenticationError(Exception):
    """Raised when authentication fails"""
    pass

# Context manager for error handling
class ErrorContext:
    """Context manager for handling errors in code blocks"""
    
    def __init__(self, context: str, user_message: str = None):
        self.context = context
        self.user_message = user_message
        self.error_handler = ErrorHandler()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error_handler.handle_exception(exc_val, self.context, self.user_message)
            return True  # Suppress the exception
        return False