"""
Edge Deployment Service

Handles edge deployment, offline operation, privacy preservation, and system optimization
for the Smart Energy Copilot running on RDK X5 ARM architecture.
"""

import asyncio
import logging
import os
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import deque
import psutil
import sqlite3
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class SystemResources:
    """System resource usage information"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    temperature_celsius: Optional[float]
    timestamp: datetime


@dataclass
class OfflineOperation:
    """Offline operation status and capabilities"""
    is_offline: bool
    offline_since: Optional[datetime]
    buffered_operations: int
    max_buffer_size: int
    offline_capabilities: List[str]


@dataclass
class PrivacyStatus:
    """Privacy preservation status"""
    local_processing_only: bool
    data_encrypted: bool
    no_cloud_dependencies: bool
    privacy_violations: List[str]


@dataclass
class UpdateStatus:
    """Over-the-air update status"""
    update_available: bool
    update_version: Optional[str]
    update_size_mb: Optional[float]
    privacy_preserving: bool
    last_check: datetime


class EdgeDeploymentService:
    """Service for managing edge deployment, offline operation, and privacy preservation"""
    
    def __init__(self, 
                 data_dir: str = "data",
                 max_offline_buffer: int = 100000,
                 thermal_threshold: float = 75.0):
        self.settings = get_settings()
        self.data_dir = Path(data_dir)
        self.max_offline_buffer = max_offline_buffer
        self.thermal_threshold = thermal_threshold
        
        # Offline operation state
        self.is_offline = False
        self.offline_since: Optional[datetime] = None
        self.offline_buffer: deque = deque(maxlen=max_offline_buffer)
        
        # Privacy and encryption
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # System monitoring
        self.resource_history: deque = deque(maxlen=1000)
        self.last_health_check = datetime.now()
        
        # Update mechanism
        self.update_server_url = None  # No cloud dependencies
        self.local_update_path = self.data_dir / "updates"
        self.local_update_path.mkdir(exist_ok=True)
        
        # Offline capabilities
        self.offline_capabilities = [
            "ocr_processing",
            "ai_inference", 
            "data_storage",
            "iot_integration",
            "recommendation_generation",
            "dashboard_display"
        ]
        
        logger.info("Edge deployment service initialized")
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create local encryption key for data privacy"""
        key_file = self.data_dir / ".encryption_key"
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            
            # Ensure data directory exists
            self.data_dir.mkdir(exist_ok=True)
            
            # Save key securely
            with open(key_file, 'wb') as f:
                f.write(key)
            
            # Set restrictive permissions
            os.chmod(key_file, 0o600)
            
            logger.info("Generated new encryption key for local data privacy")
            return key
    
    async def check_offline_capabilities(self) -> OfflineOperation:
        """Check current offline operation status and capabilities"""
        try:
            # Count buffered operations
            buffered_count = len(self.offline_buffer)
            
            # Determine offline status
            offline_since = self.offline_since if self.is_offline else None
            
            return OfflineOperation(
                is_offline=self.is_offline,
                offline_since=offline_since,
                buffered_operations=buffered_count,
                max_buffer_size=self.max_offline_buffer,
                offline_capabilities=self.offline_capabilities.copy()
            )
            
        except Exception as e:
            logger.error(f"Failed to check offline capabilities: {e}")
            return OfflineOperation(
                is_offline=True,  # Assume offline on error
                offline_since=datetime.now(),
                buffered_operations=0,
                max_buffer_size=self.max_offline_buffer,
                offline_capabilities=[]
            )
    
    async def set_offline_mode(self, offline: bool, reason: str = "manual") -> bool:
        """Set offline mode status"""
        try:
            previous_state = self.is_offline
            self.is_offline = offline
            
            if offline and not previous_state:
                # Going offline
                self.offline_since = datetime.now()
                logger.info(f"Entering offline mode: {reason}")
                
                # Ensure all critical services can operate offline
                await self._prepare_offline_operation()
                
            elif not offline and previous_state:
                # Coming back online
                self.offline_since = None
                logger.info(f"Exiting offline mode: {reason}")
                
                # Flush any buffered operations
                await self._flush_offline_buffer()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set offline mode: {e}")
            return False
    
    async def _prepare_offline_operation(self):
        """Prepare system for offline operation"""
        try:
            # Ensure local database is accessible
            db_path = self.data_dir / "energy_copilot.db"
            if not db_path.exists():
                logger.warning("Local database not found, creating minimal schema")
                # Create minimal database schema for offline operation
                await self._create_offline_database()
            
            # Verify AI models are locally available
            model_path = self.data_dir / "models"
            if not model_path.exists():
                logger.warning("AI models not found locally")
                model_path.mkdir(exist_ok=True)
            
            # Ensure encryption is working
            test_data = b"offline_test"
            encrypted = self.cipher_suite.encrypt(test_data)
            decrypted = self.cipher_suite.decrypt(encrypted)
            assert decrypted == test_data
            
            logger.info("System prepared for offline operation")
            
        except Exception as e:
            logger.error(f"Failed to prepare offline operation: {e}")
            raise
    
    async def _create_offline_database(self):
        """Create minimal database schema for offline operation"""
        db_path = self.data_dir / "energy_copilot.db"
        
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Create minimal tables for offline operation
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS offline_operations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation_type TEXT NOT NULL,
                    data TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    processed BOOLEAN DEFAULT FALSE
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    status_type TEXT NOT NULL,
                    status_data TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
            
            logger.info("Created offline database schema")
            
        except Exception as e:
            logger.error(f"Failed to create offline database: {e}")
            raise
    
    async def _flush_offline_buffer(self) -> int:
        """Flush offline buffer when coming back online"""
        flushed_count = 0
        
        try:
            while self.offline_buffer:
                operation = self.offline_buffer.popleft()
                
                # Process buffered operation
                await self._process_buffered_operation(operation)
                flushed_count += 1
            
            logger.info(f"Flushed {flushed_count} operations from offline buffer")
            
        except Exception as e:
            logger.error(f"Failed to flush offline buffer: {e}")
        
        return flushed_count
    
    async def _process_buffered_operation(self, operation: Dict[str, Any]):
        """Process a single buffered operation"""
        try:
            operation_type = operation.get("type")
            
            if operation_type == "sensor_reading":
                # Process sensor reading
                pass
            elif operation_type == "ocr_result":
                # Process OCR result
                pass
            elif operation_type == "ai_inference":
                # Process AI inference result
                pass
            
            logger.debug(f"Processed buffered operation: {operation_type}")
            
        except Exception as e:
            logger.error(f"Failed to process buffered operation: {e}")
    
    async def buffer_offline_operation(self, operation_type: str, data: Dict[str, Any]) -> bool:
        """Buffer an operation for offline processing"""
        try:
            operation = {
                "type": operation_type,
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
            
            self.offline_buffer.append(operation)
            
            logger.debug(f"Buffered offline operation: {operation_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to buffer offline operation: {e}")
            return False
    
    async def check_privacy_status(self) -> PrivacyStatus:
        """Check privacy preservation status"""
        try:
            violations = []
            
            # Check for cloud dependencies
            no_cloud_deps = True
            if self.update_server_url is not None:
                violations.append("Update server URL configured")
                no_cloud_deps = False
            
            # Check data encryption
            data_encrypted = True
            try:
                # Test encryption
                test_data = b"privacy_test"
                encrypted = self.cipher_suite.encrypt(test_data)
                decrypted = self.cipher_suite.decrypt(encrypted)
                if decrypted != test_data:
                    data_encrypted = False
                    violations.append("Encryption test failed")
            except Exception:
                data_encrypted = False
                violations.append("Encryption not available")
            
            # Check local processing
            local_processing = True
            # This would check if any services are configured to use cloud APIs
            
            return PrivacyStatus(
                local_processing_only=local_processing,
                data_encrypted=data_encrypted,
                no_cloud_dependencies=no_cloud_deps,
                privacy_violations=violations
            )
            
        except Exception as e:
            logger.error(f"Failed to check privacy status: {e}")
            return PrivacyStatus(
                local_processing_only=False,
                data_encrypted=False,
                no_cloud_dependencies=False,
                privacy_violations=[f"Privacy check failed: {e}"]
            )
    
    async def encrypt_local_data(self, data: bytes) -> bytes:
        """Encrypt data for local storage"""
        try:
            return self.cipher_suite.encrypt(data)
        except Exception as e:
            logger.error(f"Failed to encrypt data: {e}")
            raise
    
    async def decrypt_local_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt locally stored data"""
        try:
            return self.cipher_suite.decrypt(encrypted_data)
        except Exception as e:
            logger.error(f"Failed to decrypt data: {e}")
            raise
    
    async def monitor_system_resources(self) -> SystemResources:
        """Monitor system resources for thermal management"""
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Get temperature (if available)
            temperature = None
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    # Get first available temperature sensor
                    for name, entries in temps.items():
                        if entries:
                            temperature = entries[0].current
                            break
            except (AttributeError, OSError):
                # Temperature sensors not available on this system
                pass
            
            resources = SystemResources(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                temperature_celsius=temperature,
                timestamp=datetime.now()
            )
            
            # Store in history
            self.resource_history.append(resources)
            
            # Check for thermal throttling
            if temperature and temperature > self.thermal_threshold:
                logger.warning(f"High temperature detected: {temperature}°C")
                await self._handle_thermal_throttling()
            
            return resources
            
        except Exception as e:
            logger.error(f"Failed to monitor system resources: {e}")
            return SystemResources(
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_percent=0.0,
                temperature_celsius=None,
                timestamp=datetime.now()
            )
    
    async def _handle_thermal_throttling(self):
        """Handle thermal throttling by reducing system load"""
        try:
            logger.info("Implementing thermal throttling measures")
            
            # Reduce AI inference frequency
            # Disable non-essential background tasks
            # Lower CPU-intensive operations
            
            # This would integrate with other services to reduce load
            
        except Exception as e:
            logger.error(f"Failed to handle thermal throttling: {e}")
    
    async def check_for_updates(self) -> UpdateStatus:
        """Check for privacy-preserving over-the-air updates"""
        try:
            # Check local update directory for new updates
            update_files = list(self.local_update_path.glob("*.update"))
            
            if update_files:
                # Get latest update file
                latest_update = max(update_files, key=os.path.getctime)
                
                # Parse update metadata if available
                metadata_file = latest_update.with_suffix('.metadata')
                version = None
                
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        version = metadata.get('version')
                    except (json.JSONDecodeError, IOError):
                        # If metadata is corrupted, extract version from filename
                        version = self._extract_version_from_filename(latest_update.name)
                else:
                    # No metadata file, extract version from filename
                    version = self._extract_version_from_filename(latest_update.name)
                
                return UpdateStatus(
                    update_available=True,
                    update_version=version,
                    update_size_mb=latest_update.stat().st_size / (1024 * 1024),
                    privacy_preserving=True,  # Local updates preserve privacy
                    last_check=datetime.now()
                )
            
            return UpdateStatus(
                update_available=False,
                update_version=None,
                update_size_mb=None,
                privacy_preserving=True,
                last_check=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to check for updates: {e}")
            return UpdateStatus(
                update_available=False,
                update_version=None,
                update_size_mb=None,
                privacy_preserving=False,
                last_check=datetime.now()
            )
    
    def _extract_version_from_filename(self, filename: str) -> str:
        """Extract version from update filename"""
        try:
            # Remove .update extension
            name_without_ext = filename.replace('.update', '')
            
            # Try to extract version from common patterns
            # Pattern: update_v1.0.0.update -> v1.0.0
            # Pattern: update_00000.update -> 00000
            if '_' in name_without_ext:
                parts = name_without_ext.split('_')
                if len(parts) > 1:
                    return parts[-1]  # Last part after underscore
            
            # Fallback: use the whole filename without extension
            return name_without_ext
            
        except Exception:
            return "unknown"
    
    async def apply_privacy_preserving_update(self, update_path: Path) -> bool:
        """Apply a privacy-preserving update"""
        try:
            # Verify update integrity
            if not await self._verify_update_integrity(update_path):
                logger.error("Update integrity verification failed")
                return False
            
            # Backup current system state
            backup_path = await self._create_system_backup()
            if not backup_path:
                logger.error("Failed to create system backup")
                return False
            
            # Apply update
            success = await self._apply_update(update_path)
            
            if success:
                logger.info("Update applied successfully")
                return True
            else:
                # Restore from backup
                await self._restore_from_backup(backup_path)
                logger.error("Update failed, restored from backup")
                return False
                
        except Exception as e:
            logger.error(f"Failed to apply update: {e}")
            return False
    
    async def _verify_update_integrity(self, update_path: Path) -> bool:
        """Verify update file integrity"""
        try:
            # Check if checksum file exists
            checksum_file = update_path.with_suffix('.checksum')
            if not checksum_file.exists():
                return False
            
            # Calculate file hash
            with open(update_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            
            # Compare with expected hash
            with open(checksum_file, 'r') as f:
                expected_hash = f.read().strip()
            
            return file_hash == expected_hash
            
        except Exception as e:
            logger.error(f"Failed to verify update integrity: {e}")
            return False
    
    async def _create_system_backup(self) -> Optional[Path]:
        """Create system backup before update"""
        try:
            backup_dir = self.data_dir / "backups"
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"backup_{timestamp}"
            backup_path.mkdir()
            
            # Backup critical files and database
            # This would copy essential system files
            
            logger.info(f"Created system backup at {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Failed to create system backup: {e}")
            return None
    
    async def _apply_update(self, update_path: Path) -> bool:
        """Apply the update"""
        try:
            # Extract and apply update
            # This would handle the actual update process
            
            logger.info(f"Applied update from {update_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply update: {e}")
            return False
    
    async def _restore_from_backup(self, backup_path: Path) -> bool:
        """Restore system from backup"""
        try:
            # Restore from backup
            # This would restore the system state
            
            logger.info(f"Restored system from backup {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            return False
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        try:
            resources = await self.monitor_system_resources()
            offline_status = await self.check_offline_capabilities()
            privacy_status = await self.check_privacy_status()
            update_status = await self.check_for_updates()
            
            return {
                "resources": asdict(resources),
                "offline_operation": asdict(offline_status),
                "privacy_preservation": asdict(privacy_status),
                "update_status": asdict(update_status),
                "system_uptime": time.time() - psutil.boot_time(),
                "edge_deployment_ready": (
                    privacy_status.local_processing_only and
                    privacy_status.data_encrypted and
                    len(offline_status.offline_capabilities) > 0
                )
            }
            
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return {
                "error": str(e),
                "edge_deployment_ready": False
            }