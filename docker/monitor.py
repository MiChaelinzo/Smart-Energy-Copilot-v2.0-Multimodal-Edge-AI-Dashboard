#!/usr/bin/env python3
"""
System monitoring service for edge deployment.
Monitors system resources, temperature, and performance metrics.
"""

import asyncio
import logging
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
import psutil
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/app/logs/monitor.log')
    ]
)

logger = logging.getLogger(__name__)


class EdgeSystemMonitor:
    """System monitor for edge deployment environment"""
    
    def __init__(self):
        self.monitor_interval = int(os.getenv('MONITOR_INTERVAL', '30'))
        self.cpu_threshold = float(os.getenv('ALERT_THRESHOLDS_CPU', '80'))
        self.memory_threshold = float(os.getenv('ALERT_THRESHOLDS_MEMORY', '85'))
        self.temp_threshold = float(os.getenv('ALERT_THRESHOLDS_TEMP', '75'))
        
        self.running = True
        self.metrics_history = []
        self.max_history_size = 1000
        
        # Alert state tracking
        self.alert_states = {
            'cpu_high': False,
            'memory_high': False,
            'temp_high': False,
            'disk_full': False
        }
        
        logger.info(f"Edge system monitor initialized")
        logger.info(f"Monitor interval: {self.monitor_interval}s")
        logger.info(f"Thresholds - CPU: {self.cpu_threshold}%, Memory: {self.memory_threshold}%, Temp: {self.temp_threshold}°C")
    
    async def start_monitoring(self):
        """Start the monitoring loop"""
        logger.info("Starting system monitoring...")
        
        try:
            while self.running:
                # Collect system metrics
                metrics = await self.collect_system_metrics()
                
                # Store metrics in history
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history.pop(0)
                
                # Check for alerts
                await self.check_alerts(metrics)
                
                # Log metrics
                await self.log_metrics(metrics)
                
                # Wait for next interval
                await asyncio.sleep(self.monitor_interval)
                
        except asyncio.CancelledError:
            logger.info("Monitoring cancelled")
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
        finally:
            logger.info("System monitoring stopped")
    
    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics"""
        try:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu': await self._get_cpu_metrics(),
                'memory': await self._get_memory_metrics(),
                'disk': await self._get_disk_metrics(),
                'network': await self._get_network_metrics(),
                'temperature': await self._get_temperature_metrics(),
                'processes': await self._get_process_metrics(),
                'system': await self._get_system_metrics()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    async def _get_cpu_metrics(self) -> Dict[str, Any]:
        """Get CPU usage metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            
            return {
                'usage_percent': cpu_percent,
                'count': cpu_count,
                'frequency_mhz': cpu_freq.current if cpu_freq else None,
                'load_average': {
                    '1min': load_avg[0],
                    '5min': load_avg[1],
                    '15min': load_avg[2]
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get CPU metrics: {e}")
            return {'error': str(e)}
    
    async def _get_memory_metrics(self) -> Dict[str, Any]:
        """Get memory usage metrics"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                'total_mb': memory.total / (1024 * 1024),
                'available_mb': memory.available / (1024 * 1024),
                'used_mb': memory.used / (1024 * 1024),
                'usage_percent': memory.percent,
                'swap': {
                    'total_mb': swap.total / (1024 * 1024),
                    'used_mb': swap.used / (1024 * 1024),
                    'usage_percent': swap.percent
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory metrics: {e}")
            return {'error': str(e)}
    
    async def _get_disk_metrics(self) -> Dict[str, Any]:
        """Get disk usage metrics"""
        try:
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            return {
                'total_gb': disk_usage.total / (1024 * 1024 * 1024),
                'used_gb': disk_usage.used / (1024 * 1024 * 1024),
                'free_gb': disk_usage.free / (1024 * 1024 * 1024),
                'usage_percent': (disk_usage.used / disk_usage.total) * 100,
                'io': {
                    'read_bytes': disk_io.read_bytes if disk_io else 0,
                    'write_bytes': disk_io.write_bytes if disk_io else 0,
                    'read_count': disk_io.read_count if disk_io else 0,
                    'write_count': disk_io.write_count if disk_io else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get disk metrics: {e}")
            return {'error': str(e)}
    
    async def _get_network_metrics(self) -> Dict[str, Any]:
        """Get network usage metrics"""
        try:
            net_io = psutil.net_io_counters()
            net_connections = len(psutil.net_connections())
            
            return {
                'bytes_sent': net_io.bytes_sent if net_io else 0,
                'bytes_recv': net_io.bytes_recv if net_io else 0,
                'packets_sent': net_io.packets_sent if net_io else 0,
                'packets_recv': net_io.packets_recv if net_io else 0,
                'connections_count': net_connections
            }
            
        except Exception as e:
            logger.error(f"Failed to get network metrics: {e}")
            return {'error': str(e)}
    
    async def _get_temperature_metrics(self) -> Dict[str, Any]:
        """Get system temperature metrics"""
        try:
            temperatures = {}
            
            # Try to read from thermal zones
            thermal_zones = ['/sys/class/thermal/thermal_zone0/temp',
                           '/sys/class/thermal/thermal_zone1/temp']
            
            for i, zone_path in enumerate(thermal_zones):
                try:
                    if os.path.exists(zone_path):
                        with open(zone_path, 'r') as f:
                            temp_millicelsius = int(f.read().strip())
                            temp_celsius = temp_millicelsius / 1000.0
                            temperatures[f'zone_{i}'] = temp_celsius
                except:
                    continue
            
            # Try psutil sensors (if available)
            try:
                sensors = psutil.sensors_temperatures()
                for name, entries in sensors.items():
                    for i, entry in enumerate(entries):
                        temperatures[f'{name}_{i}'] = entry.current
            except (AttributeError, OSError):
                pass
            
            return temperatures
            
        except Exception as e:
            logger.error(f"Failed to get temperature metrics: {e}")
            return {'error': str(e)}
    
    async def _get_process_metrics(self) -> Dict[str, Any]:
        """Get process-related metrics"""
        try:
            process_count = len(psutil.pids())
            
            # Get top processes by CPU and memory
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Sort by CPU usage
            top_cpu = sorted(processes, key=lambda x: x.get('cpu_percent', 0), reverse=True)[:5]
            
            # Sort by memory usage
            top_memory = sorted(processes, key=lambda x: x.get('memory_percent', 0), reverse=True)[:5]
            
            return {
                'total_count': process_count,
                'top_cpu': top_cpu,
                'top_memory': top_memory
            }
            
        except Exception as e:
            logger.error(f"Failed to get process metrics: {e}")
            return {'error': str(e)}
    
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get general system metrics"""
        try:
            boot_time = psutil.boot_time()
            uptime_seconds = time.time() - boot_time
            
            return {
                'boot_time': datetime.fromtimestamp(boot_time).isoformat(),
                'uptime_seconds': uptime_seconds,
                'uptime_hours': uptime_seconds / 3600
            }
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {'error': str(e)}
    
    async def check_alerts(self, metrics: Dict[str, Any]):
        """Check metrics against thresholds and generate alerts"""
        try:
            alerts = []
            
            # Check CPU usage
            cpu_usage = metrics.get('cpu', {}).get('usage_percent', 0)
            if cpu_usage > self.cpu_threshold:
                if not self.alert_states['cpu_high']:
                    alerts.append(f"HIGH CPU USAGE: {cpu_usage:.1f}% (threshold: {self.cpu_threshold}%)")
                    self.alert_states['cpu_high'] = True
            else:
                if self.alert_states['cpu_high']:
                    alerts.append(f"CPU usage normalized: {cpu_usage:.1f}%")
                    self.alert_states['cpu_high'] = False
            
            # Check memory usage
            memory_usage = metrics.get('memory', {}).get('usage_percent', 0)
            if memory_usage > self.memory_threshold:
                if not self.alert_states['memory_high']:
                    alerts.append(f"HIGH MEMORY USAGE: {memory_usage:.1f}% (threshold: {self.memory_threshold}%)")
                    self.alert_states['memory_high'] = True
            else:
                if self.alert_states['memory_high']:
                    alerts.append(f"Memory usage normalized: {memory_usage:.1f}%")
                    self.alert_states['memory_high'] = False
            
            # Check temperature
            temperatures = metrics.get('temperature', {})
            max_temp = max(temperatures.values()) if temperatures else 0
            if max_temp > self.temp_threshold:
                if not self.alert_states['temp_high']:
                    alerts.append(f"HIGH TEMPERATURE: {max_temp:.1f}°C (threshold: {self.temp_threshold}°C)")
                    self.alert_states['temp_high'] = True
            else:
                if self.alert_states['temp_high']:
                    alerts.append(f"Temperature normalized: {max_temp:.1f}°C")
                    self.alert_states['temp_high'] = False
            
            # Check disk usage
            disk_usage = metrics.get('disk', {}).get('usage_percent', 0)
            if disk_usage > 90:  # Fixed threshold for disk
                if not self.alert_states['disk_full']:
                    alerts.append(f"DISK SPACE LOW: {disk_usage:.1f}% used")
                    self.alert_states['disk_full'] = True
            else:
                if self.alert_states['disk_full']:
                    alerts.append(f"Disk space normalized: {disk_usage:.1f}% used")
                    self.alert_states['disk_full'] = False
            
            # Log alerts
            for alert in alerts:
                logger.warning(f"ALERT: {alert}")
            
        except Exception as e:
            logger.error(f"Failed to check alerts: {e}")
    
    async def log_metrics(self, metrics: Dict[str, Any]):
        """Log system metrics"""
        try:
            # Log summary metrics
            cpu_usage = metrics.get('cpu', {}).get('usage_percent', 0)
            memory_usage = metrics.get('memory', {}).get('usage_percent', 0)
            disk_usage = metrics.get('disk', {}).get('usage_percent', 0)
            
            temperatures = metrics.get('temperature', {})
            max_temp = max(temperatures.values()) if temperatures else 0
            
            logger.info(
                f"System Status - CPU: {cpu_usage:.1f}%, "
                f"Memory: {memory_usage:.1f}%, "
                f"Disk: {disk_usage:.1f}%, "
                f"Temp: {max_temp:.1f}°C"
            )
            
            # Write detailed metrics to file
            metrics_file = '/app/logs/system_metrics.jsonl'
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')
            
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def stop_monitoring(self):
        """Stop the monitoring loop"""
        logger.info("Stopping system monitoring...")
        self.running = False


# Global monitor instance
monitor = EdgeSystemMonitor()


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    monitor.stop_monitoring()
    sys.exit(0)


async def main():
    """Main monitoring function"""
    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())