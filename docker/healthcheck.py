#!/usr/bin/env python3
"""
Health check script for edge deployment container.
Validates system health and readiness for the Smart Energy Copilot.
"""

import asyncio
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, '/app')

try:
    from src.services.edge_deployment import EdgeDeploymentService
    from src.config.settings import get_settings
except ImportError as e:
    print(f"CRITICAL: Failed to import required modules: {e}")
    sys.exit(1)


class HealthChecker:
    """Health checker for edge deployment container"""
    
    def __init__(self):
        self.data_dir = os.getenv('DATA_DIR', '/app/data')
        self.model_dir = os.getenv('MODEL_CACHE_DIR', '/app/models')
        self.edge_service = None
    
    async def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        health_status = {
            "timestamp": time.time(),
            "status": "healthy",
            "checks": {},
            "errors": []
        }
        
        try:
            # Initialize edge service
            self.edge_service = EdgeDeploymentService(
                data_dir=self.data_dir,
                max_offline_buffer=1000
            )
            
            # Check 1: Database connectivity
            health_status["checks"]["database"] = await self._check_database()
            
            # Check 2: Model availability
            health_status["checks"]["models"] = await self._check_models()
            
            # Check 3: System resources
            health_status["checks"]["resources"] = await self._check_resources()
            
            # Check 4: Privacy and encryption
            health_status["checks"]["privacy"] = await self._check_privacy()
            
            # Check 5: Offline capabilities
            health_status["checks"]["offline"] = await self._check_offline_capabilities()
            
            # Check 6: Edge deployment readiness
            health_status["checks"]["edge_ready"] = await self._check_edge_readiness()
            
            # Determine overall status
            failed_checks = [
                check_name for check_name, result in health_status["checks"].items()
                if not result.get("healthy", False)
            ]
            
            if failed_checks:
                health_status["status"] = "unhealthy"
                health_status["errors"].extend([
                    f"Failed check: {check}" for check in failed_checks
                ])
            
        except Exception as e:
            health_status["status"] = "critical"
            health_status["errors"].append(f"Health check failed: {str(e)}")
        
        return health_status
    
    async def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity and integrity"""
        try:
            db_path = Path(self.data_dir) / "energy_copilot.db"
            
            if not db_path.exists():
                return {
                    "healthy": False,
                    "message": "Database file not found",
                    "path": str(db_path)
                }
            
            # Check if database is accessible
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            conn.close()
            
            return {
                "healthy": True,
                "message": "Database accessible",
                "tables_count": len(tables)
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Database check failed: {str(e)}"
            }
    
    async def _check_models(self) -> Dict[str, Any]:
        """Check AI model availability and readiness"""
        try:
            model_path = Path(self.model_dir)
            
            if not model_path.exists():
                return {
                    "healthy": False,
                    "message": "Model directory not found",
                    "path": str(model_path)
                }
            
            # Check for required model files
            required_files = ["ernie_quantized.bin", "tokenizer.json"]
            available_files = []
            missing_files = []
            
            for file_name in required_files:
                file_path = model_path / file_name
                if file_path.exists():
                    available_files.append(file_name)
                else:
                    missing_files.append(file_name)
            
            healthy = len(missing_files) == 0
            
            return {
                "healthy": healthy,
                "message": "Models ready" if healthy else "Missing model files",
                "available_files": available_files,
                "missing_files": missing_files
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Model check failed: {str(e)}"
            }
    
    async def _check_resources(self) -> Dict[str, Any]:
        """Check system resource availability"""
        try:
            if not self.edge_service:
                return {"healthy": False, "message": "Edge service not initialized"}
            
            resources = await self.edge_service.monitor_system_resources()
            
            # Define thresholds
            cpu_threshold = 90.0
            memory_threshold = 90.0
            disk_threshold = 95.0
            temp_threshold = 80.0
            
            warnings = []
            
            if resources.cpu_percent > cpu_threshold:
                warnings.append(f"High CPU usage: {resources.cpu_percent:.1f}%")
            
            if resources.memory_percent > memory_threshold:
                warnings.append(f"High memory usage: {resources.memory_percent:.1f}%")
            
            if resources.disk_percent > disk_threshold:
                warnings.append(f"High disk usage: {resources.disk_percent:.1f}%")
            
            if resources.temperature_celsius and resources.temperature_celsius > temp_threshold:
                warnings.append(f"High temperature: {resources.temperature_celsius:.1f}°C")
            
            healthy = len(warnings) == 0
            
            return {
                "healthy": healthy,
                "message": "Resources OK" if healthy else "Resource constraints detected",
                "cpu_percent": resources.cpu_percent,
                "memory_percent": resources.memory_percent,
                "disk_percent": resources.disk_percent,
                "temperature_celsius": resources.temperature_celsius,
                "warnings": warnings
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Resource check failed: {str(e)}"
            }
    
    async def _check_privacy(self) -> Dict[str, Any]:
        """Check privacy and encryption status"""
        try:
            if not self.edge_service:
                return {"healthy": False, "message": "Edge service not initialized"}
            
            privacy_status = await self.edge_service.check_privacy_status()
            
            healthy = (
                privacy_status.local_processing_only and
                privacy_status.data_encrypted and
                privacy_status.no_cloud_dependencies and
                len(privacy_status.privacy_violations) == 0
            )
            
            return {
                "healthy": healthy,
                "message": "Privacy compliant" if healthy else "Privacy violations detected",
                "local_processing": privacy_status.local_processing_only,
                "data_encrypted": privacy_status.data_encrypted,
                "no_cloud_deps": privacy_status.no_cloud_dependencies,
                "violations": privacy_status.privacy_violations
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Privacy check failed: {str(e)}"
            }
    
    async def _check_offline_capabilities(self) -> Dict[str, Any]:
        """Check offline operation capabilities"""
        try:
            if not self.edge_service:
                return {"healthy": False, "message": "Edge service not initialized"}
            
            offline_status = await self.edge_service.check_offline_capabilities()
            
            required_capabilities = [
                "ocr_processing", "ai_inference", "data_storage",
                "iot_integration", "recommendation_generation"
            ]
            
            available_capabilities = offline_status.offline_capabilities
            missing_capabilities = [
                cap for cap in required_capabilities
                if cap not in available_capabilities
            ]
            
            healthy = len(missing_capabilities) == 0
            
            return {
                "healthy": healthy,
                "message": "Offline ready" if healthy else "Missing offline capabilities",
                "available_capabilities": available_capabilities,
                "missing_capabilities": missing_capabilities,
                "buffer_size": offline_status.buffered_operations,
                "max_buffer": offline_status.max_buffer_size
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Offline check failed: {str(e)}"
            }
    
    async def _check_edge_readiness(self) -> Dict[str, Any]:
        """Check overall edge deployment readiness"""
        try:
            if not self.edge_service:
                return {"healthy": False, "message": "Edge service not initialized"}
            
            system_health = await self.edge_service.get_system_health()
            
            edge_ready = system_health.get("edge_deployment_ready", False)
            
            return {
                "healthy": edge_ready,
                "message": "Edge deployment ready" if edge_ready else "Edge deployment not ready",
                "system_health": system_health
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Edge readiness check failed: {str(e)}"
            }


async def main():
    """Main health check function"""
    checker = HealthChecker()
    
    try:
        health_status = await checker.check_system_health()
        
        # Print health status for logging
        print(json.dumps(health_status, indent=2, default=str))
        
        # Exit with appropriate code
        if health_status["status"] == "healthy":
            print("HEALTHY: All systems operational")
            sys.exit(0)
        elif health_status["status"] == "unhealthy":
            print("UNHEALTHY: Some systems degraded")
            sys.exit(1)
        else:
            print("CRITICAL: System failure detected")
            sys.exit(2)
            
    except Exception as e:
        print(f"CRITICAL: Health check failed: {e}")
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())