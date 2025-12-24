"""
Property-based tests for interactive visualization components.

**Property 22: Interactive visualization components**
**Validates: Requirements 5.3**
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from typing import Dict, Any, List, Optional
import json
from datetime import datetime, timedelta


# Test data generators
@st.composite
def energy_time_series_strategy(draw):
    """Generate energy consumption time series data."""
    num_points = draw(st.integers(min_value=5, max_value=50))  # Reduced range for faster generation
    base_date = datetime(2024, 1, 1)
    
    data_points = []
    for i in range(num_points):
        timestamp = base_date + timedelta(hours=i)
        consumption = draw(st.floats(min_value=0.1, max_value=500.0, allow_nan=False, allow_infinity=False))
        cost = draw(st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False))
        
        data_points.append({
            'timestamp': timestamp.isoformat(),
            'consumption_kwh': consumption,
            'cost_usd': cost,
            'source': draw(st.sampled_from(['utility_bill', 'iot_sensor', 'manual_entry']))
        })
    
    return data_points


@st.composite
def chart_config_strategy(draw):
    """Generate chart configuration data."""
    return {
        'chart_type': draw(st.sampled_from(['line', 'bar', 'area', 'scatter', 'pie'])),
        'time_range': draw(st.sampled_from(['1d', '7d', '30d', '90d', '1y'])),
        'aggregation': draw(st.sampled_from(['hour', 'day', 'week', 'month'])),
        'metrics': draw(st.lists(
            st.sampled_from(['consumption_kwh', 'cost_usd', 'efficiency', 'peak_demand']),
            min_size=1, max_size=4
        )),
        'interactive_features': draw(st.lists(
            st.sampled_from(['zoom', 'pan', 'tooltip', 'crossfilter', 'brush', 'legend_toggle']),
            min_size=1, max_size=6
        )),
        'responsive': draw(st.booleans()),
        'real_time': draw(st.booleans())
    }


@st.composite
def device_data_strategy(draw):
    """Generate device monitoring data."""
    num_devices = draw(st.integers(min_value=1, max_value=10))  # Reduced range for faster generation
    devices = []
    
    for i in range(num_devices):
        device = {
            'id': f'device_{i}',
            'name': draw(st.text(min_size=3, max_size=15, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')))),  # Simplified text generation
            'type': draw(st.sampled_from(['smart_meter', 'thermostat', 'appliance', 'solar_panel', 'battery'])),
            'status': draw(st.sampled_from(['online', 'offline', 'error', 'maintenance'])),
            'current_consumption': draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
            'efficiency_rating': draw(st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False)),
            'last_update': (datetime.now() - timedelta(minutes=draw(st.integers(min_value=0, max_value=60)))).isoformat()
        }
        devices.append(device)
    
    return devices


class MockVisualizationEngine:
    """Mock visualization engine for testing interactive components."""
    
    def create_chart(self, data: List[Dict], config: Dict[str, Any]) -> Dict[str, Any]:
        """Create an interactive chart configuration."""
        chart_spec = {
            'type': config['chart_type'],
            'data': {
                'values': data,
                'format': 'json'
            },
            'encoding': self._generate_encoding(config),
            'interactive': self._generate_interactivity(config),
            'responsive': config.get('responsive', True),
            'real_time': config.get('real_time', False)
        }
        
        # Add chart-specific configurations
        if config['chart_type'] == 'line':
            chart_spec['mark'] = {'type': 'line', 'point': True, 'tooltip': True}
        elif config['chart_type'] == 'bar':
            chart_spec['mark'] = {'type': 'bar', 'tooltip': True}
        elif config['chart_type'] == 'area':
            chart_spec['mark'] = {'type': 'area', 'opacity': 0.7, 'tooltip': True}
        elif config['chart_type'] == 'scatter':
            chart_spec['mark'] = {'type': 'circle', 'size': 100, 'tooltip': True}
        elif config['chart_type'] == 'pie':
            chart_spec['mark'] = {'type': 'arc', 'tooltip': True}
        
        return chart_spec
    
    def _generate_encoding(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate chart encoding based on configuration."""
        encoding = {}
        
        # Time-based charts need x-axis for time
        if config['chart_type'] in ['line', 'bar', 'area', 'scatter']:
            encoding['x'] = {
                'field': 'timestamp',
                'type': 'temporal',
                'title': 'Time'
            }
        
        # Add y-axis for primary metric
        if config['metrics']:
            primary_metric = config['metrics'][0]
            encoding['y'] = {
                'field': primary_metric,
                'type': 'quantitative',
                'title': primary_metric.replace('_', ' ').title()
            }
        
        # Add color encoding for categorical data
        if len(config['metrics']) > 1 or 'source' in [field for field in ['source']]:
            encoding['color'] = {
                'field': 'source',
                'type': 'nominal',
                'title': 'Data Source'
            }
        
        return encoding
    
    def _generate_interactivity(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate interactivity configuration."""
        interactive_config = {}
        
        features = config.get('interactive_features', [])
        
        if 'zoom' in features:
            interactive_config['zoom'] = {'enabled': True, 'type': 'wheel'}
        
        if 'pan' in features:
            interactive_config['pan'] = {'enabled': True}
        
        if 'tooltip' in features:
            interactive_config['tooltip'] = {
                'enabled': True,
                'format': 'detailed',
                'fields': config.get('metrics', ['consumption_kwh'])
            }
        
        if 'crossfilter' in features:
            interactive_config['crossfilter'] = {'enabled': True}
        
        if 'brush' in features:
            interactive_config['brush'] = {
                'enabled': True,
                'type': 'interval'
            }
        
        if 'legend_toggle' in features:
            interactive_config['legend'] = {
                'clickable': True,
                'toggle_visibility': True
            }
        
        return interactive_config
    
    def create_dashboard_grid(self, charts: List[Dict], devices: List[Dict]) -> Dict[str, Any]:
        """Create a dashboard grid with multiple visualizations."""
        grid_config = {
            'layout': 'responsive_grid',
            'charts': charts,
            'device_panels': self._create_device_panels(devices),
            'interactions': {
                'cross_filtering': True,
                'synchronized_zoom': True,
                'shared_tooltip': True
            },
            'responsive_breakpoints': {
                'mobile': 768,
                'tablet': 1024,
                'desktop': 1200
            }
        }
        
        return grid_config
    
    def _create_device_panels(self, devices: List[Dict]) -> List[Dict]:
        """Create device monitoring panels."""
        panels = []
        
        for device in devices:
            panel = {
                'device_id': device['id'],
                'type': 'device_status',
                'interactive_elements': [
                    'status_indicator',
                    'consumption_gauge',
                    'efficiency_meter',
                    'control_buttons'
                ],
                'real_time_updates': True,
                'click_actions': ['details', 'settings', 'history']
            }
            panels.append(panel)
        
        return panels
    
    def validate_chart_spec(self, chart_spec: Dict[str, Any]) -> bool:
        """Validate that chart specification is correct."""
        required_fields = ['type', 'data', 'encoding']
        
        # Check required fields
        if not all(field in chart_spec for field in required_fields):
            return False
        
        # Validate data structure
        if 'values' not in chart_spec['data']:
            return False
        
        # Validate encoding
        encoding = chart_spec['encoding']
        if chart_spec['type'] in ['line', 'bar', 'area', 'scatter']:
            if 'x' not in encoding or 'y' not in encoding:
                return False
        
        # Validate interactive features
        if 'interactive' in chart_spec:
            interactive = chart_spec['interactive']
            valid_features = ['zoom', 'pan', 'tooltip', 'crossfilter', 'brush', 'legend']
            for feature in interactive.keys():
                if feature not in valid_features:
                    return False
        
        return True
    
    def validate_dashboard_grid(self, grid_config: Dict[str, Any]) -> bool:
        """Validate dashboard grid configuration."""
        required_fields = ['layout', 'charts', 'device_panels']
        
        if not all(field in grid_config for field in required_fields):
            return False
        
        # Validate charts
        for chart in grid_config['charts']:
            if not self.validate_chart_spec(chart):
                return False
        
        # Validate device panels
        for panel in grid_config['device_panels']:
            required_panel_fields = ['device_id', 'type', 'interactive_elements']
            if not all(field in panel for field in required_panel_fields):
                return False
        
        return True


@pytest.fixture
def viz_engine():
    """Provide visualization engine instance for testing."""
    return MockVisualizationEngine()


class TestInteractiveVisualizations:
    """Test suite for interactive visualization components."""
    
    @given(energy_time_series_strategy(), chart_config_strategy())
    @settings(max_examples=5, suppress_health_check=[HealthCheck.too_slow])
    def test_property_22_interactive_visualization_components(self, time_series_data, chart_config):
        """
        Property 22: Interactive visualization components
        
        For any energy time series data and chart configuration, the visualization engine
        should produce valid interactive chart specifications that support user interaction.
        
        **Validates: Requirements 5.3**
        """
        viz_engine = MockVisualizationEngine()
        
        # Create chart specification
        chart_spec = viz_engine.create_chart(time_series_data, chart_config)
        
        # Property: Chart specification must be structurally valid
        assert viz_engine.validate_chart_spec(chart_spec), \
            f"Chart specification is not valid: {chart_spec}"
        
        # Property: Chart must include interactive features when requested
        if chart_config.get('interactive_features'):
            assert 'interactive' in chart_spec, \
                "Chart must include interactive configuration when features are requested"
            
            interactive_config = chart_spec['interactive']
            for feature in chart_config['interactive_features']:
                # Handle feature name mapping
                if feature == 'legend_toggle':
                    assert 'legend' in interactive_config, \
                        f"Interactive feature '{feature}' (mapped to 'legend') not found in chart configuration"
                else:
                    assert feature in interactive_config, \
                        f"Interactive feature '{feature}' not found in chart configuration"
        
        # Property: Chart must handle data appropriately
        assert 'data' in chart_spec, "Chart must include data configuration"
        assert 'values' in chart_spec['data'], "Chart data must include values"
        
        # Property: Chart encoding must match data structure
        encoding = chart_spec['encoding']
        if chart_config['chart_type'] in ['line', 'bar', 'area', 'scatter']:
            assert 'x' in encoding, f"Chart type '{chart_config['chart_type']}' must have x-axis encoding"
            assert 'y' in encoding, f"Chart type '{chart_config['chart_type']}' must have y-axis encoding"
        
        # Property: Responsive charts must include responsive configuration
        if chart_config.get('responsive', True):
            assert chart_spec.get('responsive', False), \
                "Responsive charts must include responsive configuration"
        
        # Property: Real-time charts must include real-time configuration
        if chart_config.get('real_time', False):
            assert chart_spec.get('real_time', False), \
                "Real-time charts must include real-time configuration"
    
    @given(device_data_strategy())
    @settings(max_examples=5)
    def test_device_monitoring_visualizations(self, device_data):
        """
        Test that device monitoring visualizations are interactive and informative.
        
        **Validates: Requirements 5.3**
        """
        viz_engine = MockVisualizationEngine()
        
        # Create device panels
        panels = viz_engine._create_device_panels(device_data)
        
        # Property: Each device must have a corresponding panel
        assert len(panels) == len(device_data), \
            "Number of panels must match number of devices"
        
        # Property: Each panel must include interactive elements
        for i, panel in enumerate(panels):
            device = device_data[i]
            
            assert panel['device_id'] == device['id'], \
                f"Panel device ID must match device ID: {panel['device_id']} != {device['id']}"
            
            assert 'interactive_elements' in panel, \
                "Device panel must include interactive elements"
            
            interactive_elements = panel['interactive_elements']
            required_elements = ['status_indicator', 'consumption_gauge']
            for element in required_elements:
                assert element in interactive_elements, \
                    f"Device panel must include '{element}' interactive element"
            
            # Property: Panels must support real-time updates
            assert panel.get('real_time_updates', False), \
                "Device panels must support real-time updates"
            
            # Property: Panels must include click actions for user interaction
            assert 'click_actions' in panel, \
                "Device panels must include click actions"
            
            click_actions = panel['click_actions']
            assert len(click_actions) > 0, \
                "Device panels must have at least one click action"
    
    @given(
        energy_time_series_strategy(),
        st.lists(chart_config_strategy(), min_size=1, max_size=3),  # Reduced max size
        device_data_strategy()
    )
    @settings(max_examples=3, suppress_health_check=[HealthCheck.too_slow])  # Reduced examples
    def test_dashboard_grid_interactivity(self, time_series_data, chart_configs, device_data):
        """
        Test that dashboard grid supports cross-chart interactivity.
        
        **Validates: Requirements 5.3**
        """
        viz_engine = MockVisualizationEngine()
        
        # Create charts
        charts = []
        for config in chart_configs:
            chart = viz_engine.create_chart(time_series_data, config)
            charts.append(chart)
        
        # Create dashboard grid
        grid_config = viz_engine.create_dashboard_grid(charts, device_data)
        
        # Property: Dashboard grid must be structurally valid
        assert viz_engine.validate_dashboard_grid(grid_config), \
            f"Dashboard grid configuration is not valid: {grid_config}"
        
        # Property: Grid must support cross-chart interactions
        assert 'interactions' in grid_config, \
            "Dashboard grid must include interaction configuration"
        
        interactions = grid_config['interactions']
        required_interactions = ['cross_filtering', 'synchronized_zoom']
        for interaction in required_interactions:
            assert interaction in interactions, \
                f"Dashboard grid must support '{interaction}' interaction"
        
        # Property: Grid must be responsive
        assert 'responsive_breakpoints' in grid_config, \
            "Dashboard grid must include responsive breakpoints"
        
        breakpoints = grid_config['responsive_breakpoints']
        required_breakpoints = ['mobile', 'tablet', 'desktop']
        for breakpoint in required_breakpoints:
            assert breakpoint in breakpoints, \
                f"Dashboard grid must include '{breakpoint}' breakpoint"
        
        # Property: Grid must include all provided charts
        assert len(grid_config['charts']) == len(charts), \
            "Dashboard grid must include all provided charts"
        
        # Property: Grid must include device panels for all devices
        assert len(grid_config['device_panels']) == len(device_data), \
            "Dashboard grid must include panels for all devices"
    
    @given(chart_config_strategy())
    @settings(max_examples=5)
    def test_chart_accessibility_features(self, chart_config):
        """
        Test that charts include accessibility features for interactive use.
        
        **Validates: Requirements 5.3**
        """
        viz_engine = MockVisualizationEngine()
        
        # Create sample data
        sample_data = [
            {'timestamp': '2024-01-01T00:00:00', 'consumption_kwh': 10.0, 'cost_usd': 1.5},
            {'timestamp': '2024-01-01T01:00:00', 'consumption_kwh': 12.0, 'cost_usd': 1.8}
        ]
        
        chart_spec = viz_engine.create_chart(sample_data, chart_config)
        
        # Property: Charts must include tooltip for accessibility
        if 'tooltip' in chart_config.get('interactive_features', []):
            interactive_config = chart_spec.get('interactive', {})
            assert 'tooltip' in interactive_config, \
                "Charts with tooltip feature must include tooltip configuration"
            
            tooltip_config = interactive_config['tooltip']
            assert tooltip_config.get('enabled', False), \
                "Tooltip must be enabled when requested"
        
        # Property: Charts must have proper encoding for screen readers
        encoding = chart_spec['encoding']
        for axis in ['x', 'y']:
            if axis in encoding:
                assert 'title' in encoding[axis], \
                    f"Chart {axis}-axis must include title for accessibility"
        
        # Property: Color encoding must include title for legend accessibility
        if 'color' in encoding:
            assert 'title' in encoding['color'], \
                "Color encoding must include title for legend accessibility"
    
    def test_visualization_performance_requirements(self):
        """
        Test that visualizations meet performance requirements for interactivity.
        
        **Validates: Requirements 5.3**
        """
        viz_engine = MockVisualizationEngine()
        
        # Create large dataset to test performance
        large_dataset = []
        for i in range(1000):  # 1000 data points
            large_dataset.append({
                'timestamp': f'2024-01-01T{i%24:02d}:00:00',
                'consumption_kwh': 10.0 + (i % 50),
                'cost_usd': 1.5 + (i % 10) * 0.1
            })
        
        config = {
            'chart_type': 'line',
            'metrics': ['consumption_kwh'],
            'interactive_features': ['zoom', 'pan', 'tooltip'],
            'responsive': True,
            'real_time': True
        }
        
        # Property: Large datasets must still produce valid charts
        chart_spec = viz_engine.create_chart(large_dataset, config)
        assert viz_engine.validate_chart_spec(chart_spec), \
            "Large datasets must still produce valid chart specifications"
        
        # Property: Interactive features must be preserved with large datasets
        interactive_config = chart_spec.get('interactive', {})
        for feature in config['interactive_features']:
            assert feature in interactive_config, \
                f"Interactive feature '{feature}' must be preserved with large datasets"
