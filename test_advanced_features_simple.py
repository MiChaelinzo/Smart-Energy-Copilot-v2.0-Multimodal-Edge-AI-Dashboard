#!/usr/bin/env python3
"""
Simple test script to validate advanced features implementation.
"""

import asyncio
import sys
from datetime import datetime, timedelta

def test_imports():
    """Test that all advanced services can be imported."""
    print("=== Testing Advanced Features Import ===")
    
    services = [
        ('Forecasting Service', 'src.services.forecasting_service', 'EnergyForecastingService'),
        ('Smart Home Automation', 'src.services.smart_home_automation', 'SmartHomeAutomationService'),
        ('Voice Assistant', 'src.services.voice_assistant_integration', 'VoiceAssistantService'),
        ('Enterprise Features', 'src.services.enterprise_features', 'EnterpriseService'),
        ('3D Visualization', 'src.services.visualization_3d', 'Visualization3DService'),
    ]
    
    success_count = 0
    imported_services = {}
    
    for name, module, class_name in services:
        try:
            exec(f'from {module} import {class_name}')
            imported_services[name] = eval(class_name)
            print(f'[PASS] {name}: Successfully imported')
            success_count += 1
        except Exception as e:
            print(f'[FAIL] {name}: Error - {e}')
    
    print(f'\nImport Summary: {success_count}/{len(services)} services imported successfully')
    return imported_services, success_count == len(services)

def test_service_instantiation(imported_services):
    """Test that services can be instantiated."""
    print("\n=== Testing Service Instantiation ===")
    
    instances = {}
    success_count = 0
    
    for name, service_class in imported_services.items():
        try:
            instance = service_class()
            instances[name] = instance
            print(f'[PASS] {name}: Successfully instantiated')
            success_count += 1
        except Exception as e:
            print(f'[FAIL] {name}: Error - {e}')
    
    print(f'\nInstantiation Summary: {success_count}/{len(imported_services)} services instantiated successfully')
    return instances, success_count == len(imported_services)

async def test_basic_functionality(instances):
    """Test basic functionality of each service."""
    print("\n=== Testing Basic Functionality ===")
    
    success_count = 0
    total_tests = 0
    
    # Test Forecasting Service
    if 'Forecasting Service' in instances:
        try:
            forecasting = instances['Forecasting Service']
            # Test basic properties
            assert hasattr(forecasting, 'models')
            assert hasattr(forecasting, 'is_trained')
            assert forecasting.is_trained == False  # Should start untrained
            print('[PASS] Forecasting Service: Basic properties OK')
            success_count += 1
        except Exception as e:
            print(f'[FAIL] Forecasting Service: Error - {e}')
        total_tests += 1
    
    # Test Smart Home Automation
    if 'Smart Home Automation' in instances:
        try:
            smart_home = instances['Smart Home Automation']
            assert hasattr(smart_home, 'devices')
            assert hasattr(smart_home, 'scenes')
            assert len(smart_home.devices) == 0  # Should start empty
            print('[PASS] Smart Home Automation: Basic properties OK')
            success_count += 1
        except Exception as e:
            print(f'[FAIL] Smart Home Automation: Error - {e}')
        total_tests += 1
    
    # Test Voice Assistant
    if 'Voice Assistant' in instances:
        try:
            voice = instances['Voice Assistant']
            assert hasattr(voice, 'intent_handlers')
            assert hasattr(voice, 'conversation_context')
            assert len(voice.intent_handlers) > 0  # Should have handlers
            print('[PASS] Voice Assistant: Basic properties OK')
            success_count += 1
        except Exception as e:
            print(f'[FAIL] Voice Assistant: Error - {e}')
        total_tests += 1
    
    # Test Enterprise Features
    if 'Enterprise Features' in instances:
        try:
            enterprise = instances['Enterprise Features']
            assert hasattr(enterprise, 'role_permissions')
            assert hasattr(enterprise, 'api_keys')
            assert len(enterprise.role_permissions) > 0  # Should have role mappings
            print('[PASS] Enterprise Features: Basic properties OK')
            success_count += 1
        except Exception as e:
            print(f'[FAIL] Enterprise Features: Error - {e}')
        total_tests += 1
    
    # Test 3D Visualization
    if '3D Visualization' in instances:
        try:
            viz = instances['3D Visualization']
            assert hasattr(viz, 'scenes')
            assert hasattr(viz, 'ar_overlays')
            assert len(viz.scenes) == 0  # Should start empty
            print('[PASS] 3D Visualization: Basic properties OK')
            success_count += 1
        except Exception as e:
            print(f'[FAIL] 3D Visualization: Error - {e}')
        total_tests += 1
    
    print(f'\nFunctionality Summary: {success_count}/{total_tests} basic functionality tests passed')
    return success_count == total_tests

def test_mobile_app_structure():
    """Test mobile app structure."""
    print("\n=== Testing Mobile App Structure ===")
    
    import os
    
    mobile_files = [
        'mobile_app/package.json',
        'mobile_app/src/App.tsx',
        'mobile_app/src/screens/DashboardScreen.tsx',
        'mobile_app/src/services/EnergyService.ts',
        'mobile_app/src/services/DeviceService.ts',
        'mobile_app/src/services/NotificationService.ts',
        'mobile_app/src/services/VoiceService.ts',
        'mobile_app/src/components/EnergyOverviewCard.tsx',
        'mobile_app/src/context/AppContext.tsx',
    ]
    
    success_count = 0
    for file_path in mobile_files:
        if os.path.exists(file_path):
            print(f'[PASS] {file_path}: Exists')
            success_count += 1
        else:
            print(f'[FAIL] {file_path}: Missing')
    
    print(f'\nMobile App Summary: {success_count}/{len(mobile_files)} files found')
    return success_count == len(mobile_files)

def test_requirements():
    """Test that requirements.txt has all necessary dependencies."""
    print("\n=== Testing Requirements ===")
    
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
        
        required_packages = [
            'scikit-learn',
            'xgboost',
            'lightgbm',
            'redis',
            'PyJWT',
            'matplotlib',
            'plotly',
            'aiohttp',
            'websockets',
        ]
        
        success_count = 0
        for package in required_packages:
            if package in requirements:
                print(f'[PASS] {package}: Found in requirements.txt')
                success_count += 1
            else:
                print(f'[FAIL] {package}: Missing from requirements.txt')
        
        print(f'\nRequirements Summary: {success_count}/{len(required_packages)} packages found')
        return success_count == len(required_packages)
        
    except Exception as e:
        print(f'[FAIL] Error reading requirements.txt: {e}')
        return False

async def main():
    """Run all tests."""
    print("Smart Energy Copilot v3.0 - Advanced Features Validation")
    print("=" * 60)
    
    # Test imports
    imported_services, import_success = test_imports()
    
    if not import_success:
        print("\n[FAIL] Import tests failed. Cannot proceed with further testing.")
        return False
    
    # Test instantiation
    instances, instantiation_success = test_service_instantiation(imported_services)
    
    if not instantiation_success:
        print("\n[FAIL] Instantiation tests failed. Cannot proceed with functionality testing.")
        return False
    
    # Test basic functionality
    functionality_success = await test_basic_functionality(instances)
    
    # Test mobile app structure
    mobile_success = test_mobile_app_structure()
    
    # Test requirements
    requirements_success = test_requirements()
    
    # Overall summary
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    
    tests = [
        ("Import Tests", import_success),
        ("Instantiation Tests", instantiation_success),
        ("Functionality Tests", functionality_success),
        ("Mobile App Structure", mobile_success),
        ("Requirements Check", requirements_success),
    ]
    
    passed_tests = sum(1 for _, success in tests if success)
    total_tests = len(tests)
    
    for test_name, success in tests:
        status = "[PASS] PASS" if success else "[FAIL] FAIL"
        print(f"{test_name:.<30} {status}")
    
    print(f"\nOverall Result: {passed_tests}/{total_tests} test categories passed")
    
    if passed_tests == total_tests:
        print("\nALL TESTS PASSED! Smart Energy Copilot v3.0 advanced features are ready!")
        return True
    else:
        print(f"\n{total_tests - passed_tests} test categories failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)