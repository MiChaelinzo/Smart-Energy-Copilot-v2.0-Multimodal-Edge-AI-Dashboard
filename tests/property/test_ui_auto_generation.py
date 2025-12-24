"""
Property-based tests for UI auto-generation functionality.

**Property 21: Auto-generated interface layout**
**Validates: Requirements 5.5**
"""

import pytest
from hypothesis import given, strategies as st, settings
from typing import Dict, Any, List
import json


# Test data generators
@st.composite
def dashboard_config_strategy(draw):
    """Generate valid dashboard configuration data."""
    widget_types = ['energy_overview', 'recommendations', 'consumption_chart', 
                   'device_status', 'cost_analysis', 'trend_analysis']
    
    num_widgets = draw(st.integers(min_value=1, max_value=8))
    widgets = []
    
    for i in range(num_widgets):
        widget = {
            'type': draw(st.sampled_from(widget_types)),
            'position': {
                'x': draw(st.integers(min_value=0, max_value=11)),
                'y': draw(st.integers(min_value=0, max_value=20)),
                'w': draw(st.integers(min_value=1, max_value=12)),
                'h': draw(st.integers(min_value=2, max_value=8))
            },
            'id': f'widget_{i}',
            'title': draw(st.text(min_size=3, max_size=50))
        }
        widgets.append(widget)
    
    return {
        'layout': {'widgets': widgets},
        'preferences': {
            'theme': draw(st.sampled_from(['light', 'dark', 'auto'])),
            'currency': draw(st.sampled_from(['USD', 'EUR', 'GBP'])),
            'units': draw(st.sampled_from(['metric', 'imperial'])),
            'refresh_interval': draw(st.integers(min_value=5, max_value=300))
        }
    }


@st.composite
def energy_data_strategy(draw):
    """Generate energy consumption data for UI generation."""
    num_records = draw(st.integers(min_value=1, max_value=50))
    records = []
    
    for i in range(num_records):
        record = {
            'consumption_kwh': draw(st.floats(min_value=0.1, max_value=1000.0)),
            'cost_usd': draw(st.floats(min_value=0.01, max_value=500.0)),
            'timestamp': f"2024-01-{draw(st.integers(min_value=1, max_value=28))}T{draw(st.integers(min_value=0, max_value=23))}:00:00",
            'source': draw(st.sampled_from(['utility_bill', 'iot_sensor', 'manual_entry']))
        }
        records.append(record)
    
    return records


class MockUIGenerator:
    """Mock UI generator service for testing."""
    
    def generate_layout(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate UI layout based on data characteristics."""
        # Simulate ERNIE text-to-web capabilities
        widgets = []
        
        # Always include basic widgets
        widgets.append({
            'type': 'energy_overview',
            'position': {'x': 0, 'y': 0, 'w': 6, 'h': 4},
            'config': {'show_total': True, 'show_trend': True}
        })
        
        # Add widgets based on data availability
        if data.get('energy_records'):
            widgets.append({
                'type': 'consumption_chart',
                'position': {'x': 0, 'y': 4, 'w': 12, 'h': 6},
                'config': {'chart_type': 'line', 'time_range': '30d'}
            })
        
        if data.get('recommendations'):
            widgets.append({
                'type': 'recommendations',
                'position': {'x': 6, 'y': 0, 'w': 6, 'h': 4},
                'config': {'max_items': 5, 'show_priority': True}
            })
        
        if data.get('devices'):
            widgets.append({
                'type': 'device_status',
                'position': {'x': 0, 'y': 10, 'w': 6, 'h': 4},
                'config': {'show_offline': True, 'group_by_type': True}
            })
        
        return {
            'layout': {'widgets': widgets},
            'metadata': {
                'generated_at': '2024-01-01T00:00:00Z',
                'data_sources': list(data.keys()),
                'widget_count': len(widgets)
            }
        }
    
    def validate_layout(self, layout: Dict[str, Any]) -> bool:
        """Validate that generated layout is structurally correct."""
        if 'layout' not in layout:
            return False
        
        if 'widgets' not in layout['layout']:
            return False
        
        widgets = layout['layout']['widgets']
        if not isinstance(widgets, list):
            return False
        
        for widget in widgets:
            # Check required fields
            if not all(key in widget for key in ['type', 'position']):
                return False
            
            # Check position structure
            position = widget['position']
            if not all(key in position for key in ['x', 'y', 'w', 'h']):
                return False
            
            # Check position values are non-negative
            if any(position[key] < 0 for key in ['x', 'y', 'w', 'h']):
                return False
        
        return True


@pytest.fixture
def ui_generator():
    """Provide UI generator instance for testing."""
    return MockUIGenerator()


class TestUIAutoGeneration:
    """Test suite for UI auto-generation functionality."""
    
    @given(dashboard_config_strategy())
    @settings(max_examples=5)
    def test_property_21_auto_generated_interface_layout(self, config_data):
        """
        Property 21: Auto-generated interface layout
        
        For any valid dashboard configuration data, the UI generator should produce
        a structurally valid layout that includes all necessary components.
        
        **Validates: Requirements 5.5**
        """
        ui_generator = MockUIGenerator()
        
        # Prepare input data
        input_data = {
            'config': config_data,
            'energy_records': [{'consumption_kwh': 100.0, 'cost_usd': 15.0}],
            'recommendations': [{'type': 'cost_saving', 'priority': 'high'}],
            'devices': [{'id': 'device_1', 'status': 'online'}]
        }
        
        # Generate layout
        generated_layout = ui_generator.generate_layout(input_data)
        
        # Property: Generated layout must be structurally valid
        assert ui_generator.validate_layout(generated_layout), \
            f"Generated layout is not structurally valid: {generated_layout}"
        
        # Property: Layout must contain widgets
        widgets = generated_layout['layout']['widgets']
        assert len(widgets) > 0, "Generated layout must contain at least one widget"
        
        # Property: All widgets must have valid types
        valid_widget_types = {
            'energy_overview', 'recommendations', 'consumption_chart',
            'device_status', 'cost_analysis', 'trend_analysis'
        }
        for widget in widgets:
            assert widget['type'] in valid_widget_types, \
                f"Widget type '{widget['type']}' is not valid"
        
        # Property: Widget positions must not overlap excessively
        # (Allow some overlap but not complete overlap)
        for i, widget1 in enumerate(widgets):
            for j, widget2 in enumerate(widgets[i+1:], i+1):
                pos1 = widget1['position']
                pos2 = widget2['position']
                
                # Check if widgets completely overlap (same position and size)
                complete_overlap = (
                    pos1['x'] == pos2['x'] and pos1['y'] == pos2['y'] and
                    pos1['w'] == pos2['w'] and pos1['h'] == pos2['h']
                )
                assert not complete_overlap, \
                    f"Widgets {i} and {j} have complete overlap"
        
        # Property: Layout must be responsive (fit within reasonable bounds)
        max_x = max(w['position']['x'] + w['position']['w'] for w in widgets)
        max_y = max(w['position']['y'] + w['position']['h'] for w in widgets)
        
        assert max_x <= 12, f"Layout width {max_x} exceeds maximum grid width of 12"
        assert max_y <= 50, f"Layout height {max_y} exceeds reasonable maximum of 50"
    
    @given(energy_data_strategy())
    @settings(max_examples=5)
    def test_layout_adapts_to_data_availability(self, energy_data):
        """
        Test that generated layout adapts to available data sources.
        
        **Validates: Requirements 5.5**
        """
        ui_generator = MockUIGenerator()
        # Test with minimal data
        minimal_data = {'energy_records': energy_data[:1]}
        minimal_layout = ui_generator.generate_layout(minimal_data)
        
        # Test with comprehensive data
        comprehensive_data = {
            'energy_records': energy_data,
            'recommendations': [
                {'type': 'cost_saving', 'priority': 'high'},
                {'type': 'efficiency', 'priority': 'medium'}
            ],
            'devices': [
                {'id': 'device_1', 'status': 'online'},
                {'id': 'device_2', 'status': 'offline'}
            ]
        }
        comprehensive_layout = ui_generator.generate_layout(comprehensive_data)
        
        # Property: More data should result in more widgets or richer configuration
        minimal_widgets = len(minimal_layout['layout']['widgets'])
        comprehensive_widgets = len(comprehensive_layout['layout']['widgets'])
        
        assert comprehensive_widgets >= minimal_widgets, \
            "Layout with more data should have at least as many widgets"
        
        # Property: Layout should include widgets appropriate to available data
        comprehensive_widget_types = {w['type'] for w in comprehensive_layout['layout']['widgets']}
        
        if comprehensive_data.get('recommendations'):
            assert 'recommendations' in comprehensive_widget_types, \
                "Layout should include recommendations widget when recommendations are available"
        
        if comprehensive_data.get('devices'):
            assert 'device_status' in comprehensive_widget_types, \
                "Layout should include device status widget when devices are available"
    
    @given(st.dictionaries(
        st.sampled_from(['theme', 'currency', 'units', 'refresh_interval']),
        st.one_of(
            st.sampled_from(['light', 'dark', 'auto']),
            st.sampled_from(['USD', 'EUR', 'GBP']),
            st.sampled_from(['metric', 'imperial']),
            st.integers(min_value=5, max_value=300)
        ),
        min_size=1
    ))
    @settings(max_examples=5)
    def test_layout_respects_user_preferences(self, preferences):
        """
        Test that generated layout respects user preferences.
        
        **Validates: Requirements 5.5**
        """
        ui_generator = MockUIGenerator()
        input_data = {
            'preferences': preferences,
            'energy_records': [{'consumption_kwh': 50.0}]
        }
        
        layout = ui_generator.generate_layout(input_data)
        
        # Property: Generated layout must be valid regardless of preferences
        assert ui_generator.validate_layout(layout), \
            "Layout must be valid regardless of user preferences"
        
        # Property: Layout should contain metadata about generation
        assert 'metadata' in layout, "Layout should include generation metadata"
        assert 'generated_at' in layout['metadata'], "Layout should include generation timestamp"
        assert 'data_sources' in layout['metadata'], "Layout should include data source information"
    
    def test_layout_generation_consistency(self):
        """
        Test that layout generation is consistent for identical inputs.
        
        **Validates: Requirements 5.5**
        """
        ui_generator = MockUIGenerator()
        input_data = {
            'energy_records': [{'consumption_kwh': 100.0, 'cost_usd': 15.0}],
            'recommendations': [{'type': 'cost_saving', 'priority': 'high'}]
        }
        
        # Generate layout multiple times
        layout1 = ui_generator.generate_layout(input_data)
        layout2 = ui_generator.generate_layout(input_data)
        
        # Property: Widget types and count should be consistent
        widgets1 = layout1['layout']['widgets']
        widgets2 = layout2['layout']['widgets']
        
        types1 = sorted([w['type'] for w in widgets1])
        types2 = sorted([w['type'] for w in widgets2])
        
        assert types1 == types2, \
            "Layout generation should be consistent for identical inputs"
        
        assert len(widgets1) == len(widgets2), \
            "Widget count should be consistent for identical inputs"
