# app/utils/logger.py
import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import streamlit as st
from functools import wraps
import inspect
import time

class SermonAILogger:
    """Comprehensive logging system for the Sermon AI application"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.setup_loggers()
    
    def setup_loggers(self):
        """Set up specialized loggers for different aspects of the application"""
        
        # Main application logger
        self.app_logger = self._create_logger(
            'sermon_ai_app',
            f'{self.log_dir}/app_{datetime.now().strftime("%Y%m%d")}.log',
            level=logging.INFO
        )
        
        # Authentication and security logger
        self.auth_logger = self._create_logger(
            'sermon_ai_auth',
            f'{self.log_dir}/auth_{datetime.now().strftime("%Y%m%d")}.log',
            level=logging.INFO
        )
        
        # Database operations logger
        self.db_logger = self._create_logger(
            'sermon_ai_db',
            f'{self.log_dir}/database_{datetime.now().strftime("%Y%m%d")}.log',
            level=logging.INFO
        )
        
        # AI operations logger
        self.ai_logger = self._create_logger(
            'sermon_ai_models',
            f'{self.log_dir}/ai_{datetime.now().strftime("%Y%m%d")}.log',
            level=logging.INFO
        )
        
        # User activity logger
        self.activity_logger = self._create_logger(
            'sermon_ai_activity',
            f'{self.log_dir}/activity_{datetime.now().strftime("%Y%m%d")}.log',
            level=logging.INFO
        )
    
    def _create_logger(self, name: str, filename: str, level: int = logging.INFO):
        """Create a configured logger"""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(level)
        
        # Console handler for development (only warnings and above)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _get_user_context(self) -> Dict[str, Any]:
        """Get current user context for logging"""
        context = {}
        
        try:
            if hasattr(st.session_state, 'username'):
                context['username'] = st.session_state.username
            if hasattr(st.session_state, 'user_role'):
                context['user_role'] = st.session_state.user_role
            if hasattr(st.session_state, 'session_id'):
                context['session_id'] = st.session_state.session_id[:8] + '...'  # Truncated for privacy
        except:
            pass  # Streamlit session state might not be available
        
        return context
    
    def log_app_event(self, event: str, details: Dict[str, Any] = None, level: str = "info"):
        """Log general application events"""
        context = self._get_user_context()
        message = f"Event: {event}"
        
        if details:
            message += f" | Details: {json.dumps(details)}"
        if context:
            message += f" | Context: {json.dumps(context)}"
        
        getattr(self.app_logger, level.lower())(message)
    
    def log_auth_event(self, event: str, username: str = None, success: bool = True, details: Dict[str, Any] = None):
        """Log authentication events"""
        context = {'username': username or 'unknown', 'success': success}
        if details:
            context.update(details)
        
        level = "info" if success else "warning"
        message = f"Auth Event: {event} | {json.dumps(context)}"
        
        getattr(self.auth_logger, level)(message)
    
    def log_db_operation(self, operation: str, table: str = None, success: bool = True, details: Dict[str, Any] = None):
        """Log database operations"""
        context = self._get_user_context()
        context.update({
            'operation': operation,
            'table': table,
            'success': success
        })
        
        if details:
            context.update(details)
        
        level = "info" if success else "error"
        message = f"DB Operation: {operation} | {json.dumps(context)}"
        
        getattr(self.db_logger, level)(message)
    
    def log_ai_operation(self, operation: str, model: str = None, success: bool = True, 
                        response_time: float = None, details: Dict[str, Any] = None):
        """Log AI model operations"""
        context = self._get_user_context()
        context.update({
            'operation': operation,
            'model': model,
            'success': success,
            'response_time_ms': int(response_time * 1000) if response_time else None
        })
        
        if details:
            context.update(details)
        
        level = "info" if success else "error"
        message = f"AI Operation: {operation} | {json.dumps(context)}"
        
        getattr(self.ai_logger, level)(message)
    
    def log_user_activity(self, activity: str, details: Dict[str, Any] = None):
        """Log user activities"""
        context = self._get_user_context()
        context['activity'] = activity
        
        if details:
            context.update(details)
        
        message = f"User Activity: {activity} | {json.dumps(context)}"
        self.activity_logger.info(message)

# Global logger instance
logger = SermonAILogger()

# Decorators for automatic logging
def log_function_call(operation_type: str = "general"):
    """Decorator to automatically log function calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = func.__name__
            
            try:
                result = func(*args, **kwargs)
                
                # Log successful completion
                end_time = time.time()
                duration = end_time - start_time
                
                logger.log_app_event(f"function_complete", {
                    'function': function_name,
                    'operation_type': operation_type,
                    'duration_ms': int(duration * 1000)
                })
                
                return result
                
            except Exception as e:
                # Log error
                end_time = time.time()
                duration = end_time - start_time
                
                logger.log_app_event(f"function_error", {
                    'function': function_name,
                    'operation_type': operation_type,
                    'duration_ms': int(duration * 1000),
                    'error': str(e)
                }, level="error")
                
                raise
        
        return wrapper
    return decorator

def log_db_operation_decorator(operation: str):
    """Decorator for database operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                end_time = time.time()
                duration = end_time - start_time
                
                logger.log_db_operation(operation, success=True, details={
                    'function': func.__name__,
                    'duration_ms': int(duration * 1000)
                })
                
                return result
                
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                
                logger.log_db_operation(operation, success=False, details={
                    'function': func.__name__,
                    'duration_ms': int(duration * 1000),
                    'error': str(e)
                })
                
                raise
        
        return wrapper
    return decorator

def log_ai_operation_decorator(model_name: str = "unknown"):
    """Decorator for AI operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            operation = func.__name__
            
            try:
                result = func(*args, **kwargs)
                
                end_time = time.time()
                duration = end_time - start_time
                
                logger.log_ai_operation(operation, model_name, success=True, 
                                      response_time=duration)
                
                return result
                
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                
                logger.log_ai_operation(operation, model_name, success=False, 
                                      response_time=duration, details={'error': str(e)})
                
                raise
        
        return wrapper
    return decorator