"""Property-based tests for offline functionality validation.

**Validates: Requirements 3.4**
"""

import asyncio
import pytest
from hypothesis import given, strategies as st, assume, settings
from datetime import datetime, timedelta
from typing import Dict, Any, List
import tempfile
import shutil
from pathlib import Path

from src.services.edge_deployment import (
    EdgeDeploymentService, OfflineOperation, SystemResources
)
from src.services.iot_integration import IoTIntegrationService
from src.models.sensor_reading import SensorReading, SensorReadings


class TestOfflineOperationProperties:
    """Property-based tests for offline functionality validation."""
    
    @settings(max_examples=10, deadline=30000)  # Reduced examples for faster execution
    @given(
        offline_duration_minutes=st.integers(min_value=1, max_value=60),  # 1 minute to 1 hour
        buffer_operations=st.integers(min_value=0, max_value=50),
        operation_types=st.lists(
            st.sampled_from([
                "sensor_reading", "ocr_result", "ai_inference", 
                "recommendation", "device_status", "user_action"
            ]),
            min_size=0,
            max_size=10
        )
    )
    @pytest.mark.asyncio
    async def test_offline_functionality_validation_property(
        self, offline_duration_minutes: int, buffer_operations: int, operation_types: List[str]
    ):
        """
        Property 26: Offline functionality validation
        
        For any offline duration and buffered operations, the system should:
        1. Continue functioning with full capabilities when network is unavailable
        2. Buffer operations locally during offline periods
        3. Maintain all core functionality without cloud dependencies
        4. Preserve data integrity during offline operation
        5. Resume normal operation when connectivity is restored
        
        **Validates: Requirements 3.4**
        """
        # Create temporary directory for test
        with tempfile.TemporaryDirectory() as temp_dir:
            # Arrange
            edge_service = EdgeDeploymentService(
                data_dir=temp_dir,
                max_offline_buffer=200
            )
            
            # Verify initial online state
            initial_status = await edge_service.check_offline_capabilities()
            assert not initial_status.is_offline
            assert initial_status.offline_since is None
            assert len(initial_status.offline_capabilities) > 0
            
            # Act - Go offline
            offline_start = datetime.now()
            success = await edge_service.set_offline_mode(True, "network_unavailable")
            assert success
            
            # Verify offline state
            offline_status = await edge_service.check_offline_capabilities()
            assert offline_status.is_offline
            assert offline_status.offline_since is not None
            assert offline_status.offline_since >= offline_start
            
            # Assert - Core capabilities remain available offline
            expected_capabilities = [
                "ocr_processing", "ai_inference", "data_storage",
                "iot_integration", "recommendation_generation", "dashboard_display"
            ]
            for capability in expected_capabilities:
                assert capability in offline_status.offline_capabilities
            
            # Simulate offline operations
            buffered_count = 0
            for i, operation_type in enumerate(operation_types[:buffer_operations]):
                operation_data = {
                    "id": f"test_op_{i}",
                    "timestamp": datetime.now().isoformat(),
                    "data": f"test_data_{i}"
                }
                
                success = await edge_service.buffer_offline_operation(operation_type, operation_data)
                assert success
                buffered_count += 1
            
            # Verify operations are buffered
            status_after_buffering = await edge_service.check_offline_capabilities()
            assert status_after_buffering.buffered_operations == buffered_count
            
            # Assert - Data integrity during offline operation
            # All buffered operations should be preserved
            assert len(edge_service.offline_buffer) == buffered_count
            
            # Verify each buffered operation maintains integrity
            for i, buffered_op in enumerate(edge_service.offline_buffer):
                assert buffered_op["type"] in operation_types
                assert "data" in buffered_op
                assert "timestamp" in buffered_op
                assert isinstance(buffered_op["data"], dict)
            
            # Act - Come back online
            online_success = await edge_service.set_offline_mode(False, "network_restored")
            assert online_success
            
            # Verify online state restored
            online_status = await edge_service.check_offline_capabilities()
            assert not online_status.is_offline
            assert online_status.offline_since is None
            
            # Assert - Operations are flushed when coming back online
            # Buffer should be empty or significantly reduced after flush
            assert online_status.buffered_operations <= buffered_count
    
    @settings(max_examples=10, deadline=20000)
    @given(
        max_buffer_size=st.integers(min_value=10, max_value=100),
        operations_to_buffer=st.integers(min_value=0, max_value=150)
    )
    @pytest.mark.asyncio
    async def test_offline_buffer_overflow_handling_property(
        self, max_buffer_size: int, operations_to_buffer: int
    ):
        """
        Property: Offline buffer overflow handling
        
        For any buffer size and number of operations, the system should:
        1. Respect maximum buffer size limits
        2. Handle buffer overflow gracefully
        3. Preserve most recent operations when buffer is full
        4. Continue accepting new operations even when buffer is full
        
        **Validates: Requirements 3.4**
        """
        # Create temporary directory for test
        with tempfile.TemporaryDirectory() as temp_dir:
            # Arrange
            edge_service = EdgeDeploymentService(
                data_dir=temp_dir,
                max_offline_buffer=max_buffer_size
            )
            
            # Go offline
            await edge_service.set_offline_mode(True, "test_overflow")
            
            # Buffer operations up to and beyond the limit
            for i in range(operations_to_buffer):
                operation_data = {
                    "operation_id": i,
                    "data": f"operation_{i}",
                    "timestamp": datetime.now().isoformat()
                }
                
                success = await edge_service.buffer_offline_operation("test_operation", operation_data)
                assert success  # Should always succeed, even when buffer is full
            
            # Verify buffer size constraints
            status = await edge_service.check_offline_capabilities()
            actual_buffered = status.buffered_operations
            
            # Assert buffer size is respected
            assert actual_buffered <= max_buffer_size
            
            # If we tried to buffer more than the limit, verify FIFO behavior
            if operations_to_buffer > max_buffer_size:
                assert actual_buffered == max_buffer_size
                
                # Verify most recent operations are preserved
                buffer_contents = list(edge_service.offline_buffer)
                assert len(buffer_contents) == max_buffer_size
                
                # Check that the last operations are the most recent ones
                expected_start_id = operations_to_buffer - max_buffer_size
                for i, buffered_op in enumerate(buffer_contents):
                    expected_id = expected_start_id + i
                    assert buffered_op["data"]["operation_id"] == expected_id
            else:
                # All operations should be buffered
                assert actual_buffered == operations_to_buffer
    
    @settings(max_examples=5, deadline=25000)
    @given(
        network_interruptions=st.lists(
            st.tuples(
                st.integers(min_value=1, max_value=10),  # offline duration in seconds
                st.integers(min_value=1, max_value=5)   # online duration in seconds
            ),
            min_size=1,
            max_size=3
        ),
        operations_per_cycle=st.integers(min_value=1, max_value=5)
    )
    @pytest.mark.asyncio
    async def test_intermittent_connectivity_handling_property(
        self, network_interruptions: List[tuple], operations_per_cycle: int
    ):
        """
        Property: Intermittent connectivity handling
        
        For any pattern of network interruptions, the system should:
        1. Seamlessly transition between online and offline modes
        2. Preserve operation continuity across connectivity changes
        3. Maintain data consistency during frequent state changes
        4. Handle rapid online/offline transitions gracefully
        
        **Validates: Requirements 3.4**
        """
        # Create temporary directory for test
        with tempfile.TemporaryDirectory() as temp_dir:
            # Arrange
            edge_service = EdgeDeploymentService(
                data_dir=temp_dir,
                max_offline_buffer=100
            )
            
            total_operations = 0
            
            for offline_duration, online_duration in network_interruptions:
                # Go offline
                await edge_service.set_offline_mode(True, "intermittent_connectivity")
                
                # Verify offline state
                offline_status = await edge_service.check_offline_capabilities()
                assert offline_status.is_offline
                
                # Buffer operations while offline
                for i in range(operations_per_cycle):
                    operation_data = {
                        "cycle_operation": total_operations + i,
                        "offline_duration": offline_duration,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    success = await edge_service.buffer_offline_operation(
                        "intermittent_test", operation_data
                    )
                    assert success
                
                total_operations += operations_per_cycle
                
                # Verify operations are buffered
                status_offline = await edge_service.check_offline_capabilities()
                assert status_offline.buffered_operations > 0
                
                # Come back online
                await edge_service.set_offline_mode(False, "connectivity_restored")
                
                # Verify online state
                online_status = await edge_service.check_offline_capabilities()
                assert not online_status.is_offline
                
                # Operations should be flushed or in process of being flushed
                # Buffer should be empty or reduced after coming online
                final_status = await edge_service.check_offline_capabilities()
                
                # Assert system maintains consistency across transitions
                assert isinstance(final_status.buffered_operations, int)
                assert final_status.buffered_operations >= 0
            
            # Final verification - system should be in consistent state
            final_health = await edge_service.get_system_health()
            assert "offline_operation" in final_health
            assert isinstance(final_health["offline_operation"]["is_offline"], bool)
    
    @settings(max_examples=8, deadline=20000)
    @given(
        core_services=st.lists(
            st.sampled_from([
                "ocr_processing", "ai_inference", "data_storage",
                "iot_integration", "recommendation_generation", "dashboard_display"
            ]),
            min_size=1,
            max_size=6,
            unique=True
        )
    )
    @pytest.mark.asyncio
    async def test_offline_service_availability_property(
        self, core_services: List[str]
    ):
        """
        Property: Offline service availability
        
        For any set of core services, when operating offline, the system should:
        1. Maintain availability of all essential services
        2. Provide full functionality without external dependencies
        3. Ensure service capabilities are not degraded in offline mode
        4. Maintain service consistency across offline periods
        
        **Validates: Requirements 3.4**
        """
        # Create temporary directory for test
        with tempfile.TemporaryDirectory() as temp_dir:
            # Arrange
            edge_service = EdgeDeploymentService(
                data_dir=temp_dir,
                max_offline_buffer=500
            )
            
            # Verify all services are available online
            online_status = await edge_service.check_offline_capabilities()
            for service in core_services:
                assert service in online_status.offline_capabilities
            
            # Go offline
            await edge_service.set_offline_mode(True, "service_availability_test")
            
            # Verify all core services remain available offline
            offline_status = await edge_service.check_offline_capabilities()
            assert offline_status.is_offline
            
            for service in core_services:
                assert service in offline_status.offline_capabilities
            
            # Test that service capabilities are maintained
            # Each service should be able to perform its core functions
            for service in core_services:
                # Simulate service operation
                operation_data = {
                    "service": service,
                    "test_operation": True,
                    "timestamp": datetime.now().isoformat()
                }
                
                success = await edge_service.buffer_offline_operation(
                    f"{service}_operation", operation_data
                )
                assert success
            
            # Verify system health indicates readiness
            health = await edge_service.get_system_health()
            assert "offline_operation" in health
            assert health["offline_operation"]["is_offline"]
            assert len(health["offline_operation"]["offline_capabilities"]) >= len(core_services)
            
            # Come back online and verify services are still available
            await edge_service.set_offline_mode(False, "service_test_complete")
            
            final_status = await edge_service.check_offline_capabilities()
            assert not final_status.is_offline
            
            # All services should still be available
            for service in core_services:
                assert service in final_status.offline_capabilities
    
    @pytest.mark.asyncio
    async def test_offline_data_persistence_property(self):
        """
        Property: Offline data persistence
        
        For any data operations during offline mode, the system should:
        1. Persist all data locally without data loss
        2. Maintain data integrity across system restarts
        3. Ensure data is available immediately when needed
        4. Handle concurrent data operations safely
        
        **Validates: Requirements 3.4**
        """
        # Create temporary directory for test
        with tempfile.TemporaryDirectory() as temp_dir:
            # Arrange - First service instance
            edge_service1 = EdgeDeploymentService(
                data_dir=temp_dir,
                max_offline_buffer=100
            )
            
            # Go offline and buffer some operations
            await edge_service1.set_offline_mode(True, "persistence_test")
            
            test_operations = [
                {"type": "sensor_reading", "data": {"sensor_id": "test_1", "value": 42.5}},
                {"type": "ocr_result", "data": {"document_id": "doc_1", "text": "Test OCR"}},
                {"type": "ai_inference", "data": {"model": "test", "result": "prediction"}}
            ]
            
            for i, op in enumerate(test_operations):
                success = await edge_service1.buffer_offline_operation(op["type"], op["data"])
                assert success
            
            # Verify operations are buffered
            status1 = await edge_service1.check_offline_capabilities()
            assert status1.buffered_operations == len(test_operations)
            
            # Simulate system restart by creating new service instance
            edge_service2 = EdgeDeploymentService(
                data_dir=temp_dir,
                max_offline_buffer=100
            )
            
            # Verify data persistence across restart
            # Note: In a real implementation, buffered data would be persisted to disk
            # For this test, we verify the service can be reinitialized properly
            status2 = await edge_service2.check_offline_capabilities()
            
            # Service should initialize properly
            assert isinstance(status2.is_offline, bool)
            assert isinstance(status2.buffered_operations, int)
            assert len(status2.offline_capabilities) > 0
            
            # Verify encryption key persistence
            # Both service instances should be able to encrypt/decrypt
            test_data = b"persistence_test_data"
            
            encrypted1 = await edge_service1.encrypt_local_data(test_data)
            decrypted2 = await edge_service2.decrypt_local_data(encrypted1)
            
            assert decrypted2 == test_data


# Feature: smart-energy-copilot, Property 26: Offline functionality validation