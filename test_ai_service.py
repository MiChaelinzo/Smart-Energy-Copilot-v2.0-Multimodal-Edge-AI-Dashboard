"""
Simple test script for ERNIE AI service functionality.
"""

import asyncio
from datetime import datetime, timedelta
from src.services.ai_service import get_ai_service
from src.models.energy_consumption import EnergyConsumption
from src.models.sensor_reading import SensorReading


async def test_ai_service():
    """Test the AI service functionality."""
    print("Testing ERNIE AI Service...")
    
    try:
        # Get AI service
        ai_service = await get_ai_service()
        print(f"✓ AI service initialized: {ai_service.model_manager.is_loaded()}")
        
        # Force model loading if not loaded
        if not ai_service.model_manager.is_loaded():
            print("Model not loaded, attempting to load...")
            result = await ai_service.model_manager.load_model()
            print(f"Load result: {result}")
        print(f"✓ Model loaded: {ai_service.model_manager.is_loaded()}")
        print(f"Model manager _model_loaded flag: {ai_service.model_manager._model_loaded}")
        
        # Create test energy consumption data
        test_consumption = [
            EnergyConsumption(
                id="test_1",
                timestamp=datetime.now() - timedelta(hours=i),
                source="utility_bill",
                consumption_kwh=10.5 + i * 0.5,
                cost_usd=2.1 + i * 0.1,
                billing_period={
                    'start_date': datetime.now() - timedelta(days=30),
                    'end_date': datetime.now()
                },
                confidence_score=0.9
            )
            for i in range(24)  # 24 hours of data
        ]
        
        # Test pattern analysis
        print("\nTesting pattern analysis...")
        patterns = await ai_service.pattern_analyzer.identify_patterns(test_consumption)
        print(f"✓ Identified {len(patterns)} patterns")
        
        for pattern in patterns:
            print(f"  - {pattern.pattern_type}: {pattern.description} (confidence: {pattern.confidence:.2f})")
        
        # Test real-time inference
        print("\nTesting real-time inference...")
        result = await ai_service.real_time_inference(
            test_consumption[0],
            inference_type="pattern_analysis"
        )
        print(f"✓ Real-time inference completed in {result['processing_time_ms']:.2f}ms")
        print(f"  Confidence: {result['confidence']:.2f}")
        
        # Test text analysis
        print("\nTesting text analysis...")
        text_result = await ai_service.real_time_inference(
            "Energy consumption increased by 15% this month due to heating",
            inference_type="text_analysis"
        )
        print(f"✓ Text analysis completed in {text_result['processing_time_ms']:.2f}ms")
        print(f"  Energy relevance: {text_result['result'].get('energy_relevance', 0):.2f}")
        
        # Test batch inference
        print("\nTesting batch inference...")
        batch_result = await ai_service.batch_inference(
            test_consumption[:10],
            inference_type="comprehensive_analysis"
        )
        print(f"✓ Batch inference completed in {batch_result['processing_time_ms']:.2f}ms")
        print(f"  Processed {batch_result['batch_size']} items")
        print(f"  Found {batch_result['summary']['total_patterns']} patterns")
        
        print("\n🎉 All AI service tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_ai_service())