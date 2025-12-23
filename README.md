# Smart Energy Copilot v2.0

AI-powered energy optimization dashboard for edge deployment on RDK X5 devices.

## Features

- **OCR Document Processing**: Extract energy data from utility bills using PaddleOCR-VL
- **AI-Powered Analysis**: Fine-tuned ERNIE model for energy pattern analysis
- **Multi-Agent System**: CAMEL-AI orchestration for comprehensive recommendations
- **IoT Integration**: Support for MQTT, HTTP REST, and Modbus protocols
- **Edge Deployment**: Optimized for local processing on RDK X5 hardware
- **Privacy-First**: All data processing occurs locally, no cloud dependencies

## Quick Start

### Development Setup

1. Clone the repository
2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy environment configuration:
   ```bash
   cp .env.example .env
   ```
5. Initialize database:
   ```bash
   alembic upgrade head
   ```
6. Run the application:
   ```bash
   python src/main.py
   ```

### Docker Deployment

1. Build the container:
   ```bash
   docker-compose build
   ```
2. Start the services:
   ```bash
   docker-compose up -d
   ```

## Project Structure

```
├── src/
│   ├── components/          # UI and interface components
│   ├── services/           # Business logic and external integrations
│   ├── models/             # Data models and schemas
│   ├── database/           # Database connection and utilities
│   ├── config/             # Configuration and logging
│   └── main.py             # Application entry point
├── tests/
│   ├── unit/               # Unit tests
│   └── property/           # Property-based tests
├── migrations/             # Database migrations
├── data/                   # Local data storage
├── logs/                   # Application logs
└── models/                 # AI model files
```

## Testing

Run all tests:
```bash
pytest
```

Run specific test types:
```bash
pytest -m unit          # Unit tests only
pytest -m property      # Property-based tests only
```

## License

See LICENSE file for details.