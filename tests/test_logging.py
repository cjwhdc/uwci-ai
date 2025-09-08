# tests/test_logging.py
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
