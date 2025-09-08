# tests/test_error_handling.py
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
