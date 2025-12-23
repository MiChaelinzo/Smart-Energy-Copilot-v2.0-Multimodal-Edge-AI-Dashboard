# Smart Energy Copilot v2.0 Design Document

## Overview

The Smart Energy Copilot v2.0 is a comprehensive multimodal AI system that transforms energy consumption data into actionable optimization insights. The system combines document processing, AI-powered analysis, multi-agent orchestration, and edge deployment to provide real-time energy optimization recommendations while maintaining complete data privacy through local processing.

The architecture follows a modular design with clear separation between data ingestion, AI processing, agent coordination, and user interface layers. All components are designed to run efficiently on edge hardware while providing enterprise-grade reliability and performance.

## Architecture

The system employs a layered architecture optimized for edge deployment:

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Dashboard Layer                      │
│  (Auto-generated UI, Visualizations, User Interactions)    │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                 Multi-Agent Orchestration                   │
│     (CAMEL-AI: Efficiency, Cost, Environmental Agents)     │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                   AI Processing Engine                      │
│        (Fine-tuned ERNIE, Pattern Analysis, ML Models)     │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                   Data Processing Layer                     │
│    (OCR Engine, IoT Integration, Data Validation)          │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                     Storage & Cache                         │
│      (Local SQLite, Time-series DB, Model Cache)           │
└─────────────────────────────────────────────────────────────┘
```

The architecture ensures data flows efficiently from raw inputs (documents, IoT sensors) through AI processing to actionable recommendations, with all processing occurring locally on the RDK X5 edge device.

## Components and Interfaces

### OCR Processing Engine
- **Technology**: PaddleOCR-VL for document text extraction and layout analysis
- **Input**: Utility bills (PDF, JPEG, PNG), energy reports, IoT sensor logs
- **Output**: Structured energy data with confidence scores
- **Interface**: REST API for document upload and processing status
- **Key Features**: Multi-format support, confidence scoring, error handling for poor quality images

### ERNIE AI Model
- **Technology**: Fine-tuned ERNIE multimodal model using Unsloth optimization
- **Training Data**: Energy consumption patterns, optimization strategies, domain-specific terminology
- **Input**: Structured energy data, historical patterns, IoT sensor readings
- **Output**: Energy insights, pattern analysis, preliminary recommendations
- **Interface**: Python API with batch and real-time inference capabilities
- **Optimization**: Quantized for edge deployment, <2GB memory footprint

### CAMEL Multi-Agent System
- **Agents**:
  - **Efficiency Advisor**: Focuses on energy waste reduction and device optimization
  - **Cost Forecaster**: Analyzes pricing patterns and cost-saving opportunities
  - **Eco-Friendly Planner**: Provides environmental impact recommendations
- **Coordination**: CAMEL-AI framework managing agent collaboration and consensus
- **Interface**: Agent communication protocol with conflict resolution
- **Output**: Synthesized recommendations with agent attribution

### IoT Integration Layer
- **Protocols**: MQTT, HTTP REST, Modbus for device communication
- **Data Sources**: Smart meters, temperature sensors, occupancy detectors, appliance monitors
- **Processing**: Real-time data validation, interpolation, anomaly detection
- **Interface**: Standardized sensor data API with device auto-discovery
- **Reliability**: Offline operation capability, data buffering, reconnection handling

### Edge Deployment Runtime
- **Platform**: RDK X5 ARM-based edge computing device
- **Container**: Docker-based deployment for easy updates and isolation
- **Resource Management**: CPU/GPU scheduling, memory optimization, thermal management
- **Security**: Local data encryption, secure model updates, access control
- **Monitoring**: System health, performance metrics, error logging

### Web Dashboard
- **Technology**: Auto-generated using ERNIE text-to-web capabilities
- **Features**: Interactive charts, recommendation cards, progress tracking, device status
- **Interface**: RESTful API for data retrieval and user actions
- **Responsiveness**: Mobile-friendly design, real-time updates via WebSocket
- **Customization**: User preferences, dashboard layout options, alert settings

## Data Models

### Energy Consumption Record
```typescript
interface EnergyConsumption {
  id: string;
  timestamp: Date;
  source: 'utility_bill' | 'iot_sensor' | 'manual_entry';
  consumption_kwh: number;
  cost_usd: number;
  billing_period: {
    start_date: Date;
    end_date: Date;
  };
  device_breakdown?: DeviceConsumption[];
  confidence_score: number;
  raw_data?: any;
}
```

### Device Consumption
```typescript
interface DeviceConsumption {
  device_id: string;
  device_type: string;
  consumption_kwh: number;
  efficiency_rating?: string;
  usage_hours: number;
  estimated_cost: number;
}
```

### Optimization Recommendation
```typescript
interface OptimizationRecommendation {
  id: string;
  type: 'cost_saving' | 'efficiency' | 'environmental';
  priority: 'high' | 'medium' | 'low';
  title: string;
  description: string;
  implementation_steps: string[];
  estimated_savings: {
    annual_cost_usd: number;
    annual_kwh: number;
    co2_reduction_kg: number;
  };
  difficulty: 'easy' | 'moderate' | 'complex';
  agent_source: string;
  confidence: number;
  created_at: Date;
  status: 'pending' | 'implemented' | 'dismissed';
}
```

### IoT Sensor Reading
```typescript
interface SensorReading {
  sensor_id: string;
  device_type: string;
  timestamp: Date;
  readings: {
    power_watts?: number;
    voltage?: number;
    current_amps?: number;
    temperature_celsius?: number;
    humidity_percent?: number;
    occupancy?: boolean;
  };
  quality_score: number;
  location: string;
}
```

### Energy Pattern
```typescript
interface EnergyPattern {
  id: string;
  pattern_type: 'daily' | 'weekly' | 'seasonal' | 'anomaly';
  description: string;
  time_range: {
    start: Date;
    end: Date;
  };
  consumption_trend: 'increasing' | 'decreasing' | 'stable';
  peak_hours: number[];
  average_consumption: number;
  cost_impact: number;
  confidence: number;
}
```
