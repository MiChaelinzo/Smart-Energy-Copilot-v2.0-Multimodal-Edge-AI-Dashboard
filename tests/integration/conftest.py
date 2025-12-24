"""
Configuration and fixtures for integration tests.
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import patch, AsyncMock

# Configure pytest for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
def mock_database():
    """Mock database connections for all integration tests."""
    with patch('src.database.connection.get_db_session') as mock_db:
        mock_session = AsyncMock()
        mock_db.return_value.__aenter__.return_value = mock_session
        mock_db.return_value.__aexit__.return_value = None
        
        # Mock common database operations
        mock_session.add.return_value = None
        mock_session.commit.return_value = None
        mock_session.execute.return_value.fetchone.return_value = (1,)
        mock_session.query.return_value.filter.return_value.count.return_value = 0
        mock_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
        
        yield mock_session

@pytest.fixture(autouse=True)
def mock_file_operations():
    """Mock file system operations for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('tempfile.gettempdir', return_value=temp_dir):
            yield temp_dir

@pytest.fixture
def integration_test_config():
    """Configuration for integration tests."""
    return {
        'test_timeout': 30,
        'max_concurrent_requests': 20,
        'performance_threshold_ms': 500,
        'success_rate_threshold': 0.8,
        'confidence_threshold': 0.75
    }