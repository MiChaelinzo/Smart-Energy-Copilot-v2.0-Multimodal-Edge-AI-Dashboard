#!/usr/bin/env python3
"""Initialize database schema."""

import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from database.connection import engine, Base
from config.logging import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


def init_database():
    """Initialize the database schema."""
    try:
        # Create data directory if it doesn't exist
        data_dir = project_root / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        logger.info("Database initialized successfully")
        print("✅ Database initialized successfully")
        
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        print(f"❌ Failed to initialize database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    init_database()