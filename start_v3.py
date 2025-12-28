#!/usr/bin/env python3
"""
Smart Energy Copilot v3.0 Startup Script
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def start_advanced_services():
    """Start all advanced services."""
    print("Starting Smart Energy Copilot v3.0...")
    
    # Import services
    from src.services.forecasting_service import forecasting_service
    from src.services.smart_home_automation import smart_home_service
    from src.services.voice_assistant_integration import voice_assistant_service
    from src.services.enterprise_features import enterprise_service
    from src.services.visualization_3d import visualization_3d_service
    
    # Start services
    services_started = []
    
    try:
        # Start smart home automation
        if await smart_home_service.start_service():
            services_started.append("Smart Home Automation")
            print("[PASS] Smart Home Automation Service started")
        
        # Start voice assistant
        if await voice_assistant_service.start_service():
            services_started.append("Voice Assistant")
            print("[PASS] Voice Assistant Service started")
        
        # Initialize enterprise features (if enabled)
        if await enterprise_service.initialize():
            services_started.append("Enterprise Features")
            print("[PASS] Enterprise Features initialized")
        
        # Initialize 3D visualization
        if await visualization_3d_service.initialize():
            services_started.append("3D Visualization")
            print("[PASS] 3D Visualization Service initialized")
        
        print(f"\nStarted {len(services_started)} advanced services:")
        for service in services_started:
            print(f"   • {service}")
        
        # Start main application
        from src.main import app
        import uvicorn
        
        print("\nStarting web server on http://localhost:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000)
        
    except Exception as e:
        print(f"[FAIL] Error starting services: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(start_advanced_services())
