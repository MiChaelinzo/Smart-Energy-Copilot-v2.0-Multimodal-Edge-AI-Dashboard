"""Database connection and session management."""

import os
import asyncio
from contextlib import asynccontextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/energy_copilot.db")

# Create engine with SQLite optimizations for time-series data
engine = create_engine(
    DATABASE_URL,
    poolclass=StaticPool,
    connect_args={
        "check_same_thread": False,
        "timeout": 20,
    },
    echo=os.getenv("SQL_DEBUG", "false").lower() == "true"
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@asynccontextmanager
async def get_db_session():
    """Get async database session (wrapper around sync session)."""
    # For now, use a simple wrapper around sync session
    # In production, consider using async SQLAlchemy with aiosqlite
    db = SessionLocal()
    try:
        # Create a simple async wrapper
        class AsyncSessionWrapper:
            def __init__(self, session):
                self._session = session
            
            async def execute(self, query):
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self._session.execute, query)
            
            def add(self, obj):
                return self._session.add(obj)
            
            async def commit(self):
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self._session.commit)
            
            async def close(self):
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self._session.close)
        
        yield AsyncSessionWrapper(db)
    finally:
        db.close()