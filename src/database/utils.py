"""Database utilities for migration, backup, and optimization."""

import os
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional
from sqlalchemy import text, inspect
from sqlalchemy.engine import Engine
from src.database.connection import engine, Base
from src.config.logging import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """Database management utilities."""
    
    def __init__(self, engine: Engine = engine):
        self.engine = engine
        self.db_path = self._get_db_path()
    
    def _get_db_path(self) -> Optional[str]:
        """Extract database file path from engine URL."""
        url = str(self.engine.url)
        if url.startswith('sqlite:///'):
            return url[10:]  # Remove 'sqlite:///' prefix
        return None
    
    def create_tables(self) -> None:
        """Create all database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def drop_tables(self) -> None:
        """Drop all database tables."""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise
    
    def backup_database(self, backup_path: Optional[str] = None) -> str:
        """Create a backup of the database."""
        if not self.db_path or not os.path.exists(self.db_path):
            raise ValueError("Database file not found")
        
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = Path(self.db_path).parent / "backups"
            backup_dir.mkdir(exist_ok=True)
            backup_path = str(backup_dir / f"energy_copilot_backup_{timestamp}.db")
        
        try:
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backed up to: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Failed to backup database: {e}")
            raise
    
    def restore_database(self, backup_path: str) -> None:
        """Restore database from backup."""
        if not os.path.exists(backup_path):
            raise ValueError(f"Backup file not found: {backup_path}")
        
        if not self.db_path:
            raise ValueError("Database path not configured")
        
        try:
            # Close all connections
            self.engine.dispose()
            
            # Restore backup
            shutil.copy2(backup_path, self.db_path)
            logger.info(f"Database restored from: {backup_path}")
        except Exception as e:
            logger.error(f"Failed to restore database: {e}")
            raise
    
    def optimize_database(self) -> None:
        """Optimize database for time-series data performance."""
        if not self.db_path:
            logger.warning("Cannot optimize non-file database")
            return
        
        try:
            with self.engine.connect() as conn:
                # Enable WAL mode for better concurrent access
                conn.execute(text("PRAGMA journal_mode=WAL"))
                
                # Optimize for time-series queries
                conn.execute(text("PRAGMA synchronous=NORMAL"))
                conn.execute(text("PRAGMA cache_size=10000"))
                conn.execute(text("PRAGMA temp_store=MEMORY"))
                
                # Create indexes for time-series queries
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_energy_consumption_timestamp 
                    ON energy_consumption(timestamp)
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_energy_consumption_source_timestamp 
                    ON energy_consumption(source, timestamp)
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_sensor_readings_sensor_timestamp 
                    ON sensor_readings(sensor_id, timestamp)
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_sensor_readings_location_timestamp 
                    ON sensor_readings(location, timestamp)
                """))
                
                # Analyze tables for query optimization
                conn.execute(text("ANALYZE"))
                
                conn.commit()
                logger.info("Database optimized for time-series operations")
        except Exception as e:
            logger.error(f"Failed to optimize database: {e}")
            raise
    
    def get_database_info(self) -> dict:
        """Get database information and statistics."""
        try:
            with self.engine.connect() as conn:
                # Get table information
                inspector = inspect(self.engine)
                tables = inspector.get_table_names()
                
                info = {
                    "database_path": self.db_path,
                    "tables": tables,
                    "table_stats": {}
                }
                
                # Get row counts for each table
                for table in tables:
                    try:
                        result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        count = result.scalar()
                        info["table_stats"][table] = {"row_count": count}
                    except Exception as e:
                        info["table_stats"][table] = {"error": str(e)}
                
                # Get database size if file-based
                if self.db_path and os.path.exists(self.db_path):
                    size_bytes = os.path.getsize(self.db_path)
                    info["size_mb"] = round(size_bytes / (1024 * 1024), 2)
                
                return info
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            raise
    
    def vacuum_database(self) -> None:
        """Vacuum database to reclaim space and optimize performance."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("VACUUM"))
                logger.info("Database vacuumed successfully")
        except Exception as e:
            logger.error(f"Failed to vacuum database: {e}")
            raise
    
    def check_database_integrity(self) -> bool:
        """Check database integrity."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("PRAGMA integrity_check"))
                integrity_result = result.fetchone()
                
                if integrity_result and integrity_result[0] == "ok":
                    logger.info("Database integrity check passed")
                    return True
                else:
                    logger.error(f"Database integrity check failed: {integrity_result}")
                    return False
        except Exception as e:
            logger.error(f"Failed to check database integrity: {e}")
            return False


def initialize_database() -> None:
    """Initialize database with tables and optimizations."""
    db_manager = DatabaseManager()
    
    # Create data directory if it doesn't exist
    if db_manager.db_path:
        data_dir = Path(db_manager.db_path).parent
        data_dir.mkdir(exist_ok=True)
    
    # Create tables
    db_manager.create_tables()
    
    # Optimize for time-series operations
    db_manager.optimize_database()
    
    logger.info("Database initialized successfully")


def backup_database(backup_path: Optional[str] = None) -> str:
    """Convenience function to backup database."""
    db_manager = DatabaseManager()
    return db_manager.backup_database(backup_path)


def restore_database(backup_path: str) -> None:
    """Convenience function to restore database."""
    db_manager = DatabaseManager()
    db_manager.restore_database(backup_path)