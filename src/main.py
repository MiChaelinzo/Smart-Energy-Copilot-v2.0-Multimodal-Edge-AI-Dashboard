"""Smart Energy Copilot v2.0 - Main application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config.settings import settings
from src.config.logging import setup_logging, get_logger
from src.components.ocr_api import router as ocr_router
from src.components.ai_api import router as ai_router
from src.components.health_api import router as health_router
from src.components.alerts_api import router as alerts_router
from src.services.system_monitor import system_monitor
from src.services.energy_alerts import energy_alerts_service

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Smart Energy Copilot v3.1",
    description="AI-powered energy optimization dashboard for edge deployment with real-time alerts and dark mode support",
    version="3.1.0",
    debug=settings.debug
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(ocr_router)
app.include_router(ai_router)
app.include_router(health_router)
app.include_router(alerts_router)


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("Starting Smart Energy Copilot v2.0", 
                environment=settings.environment,
                debug=settings.debug)
    
    # Start system monitoring
    try:
        await system_monitor.start_monitoring()
        logger.info("System monitoring started successfully")
    except Exception as e:
        logger.error(f"Failed to start system monitoring: {e}")
    
    # Start energy alerts service
    try:
        await energy_alerts_service.start_service()
        logger.info("Energy alerts service started successfully")
    except Exception as e:
        logger.error(f"Failed to start energy alerts service: {e}")
    
    # Initialize AI service
    try:
        from src.services.ai_service import get_ai_service
        ai_service = await get_ai_service()
        logger.info("AI service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AI service: {e}")
        # Continue startup even if AI service fails to initialize


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("Shutting down Smart Energy Copilot v2.0")
    
    # Stop system monitoring
    try:
        await system_monitor.stop_monitoring()
        logger.info("System monitoring stopped")
    except Exception as e:
        logger.error(f"Error stopping system monitoring: {e}")
    
    # Stop energy alerts service
    try:
        await energy_alerts_service.stop_service()
        logger.info("Energy alerts service stopped")
    except Exception as e:
        logger.error(f"Error stopping energy alerts service: {e}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Smart Energy Copilot v3.1",
        "version": "3.1.0",
        "status": "running",
        "features": ["energy_alerts", "dark_mode", "forecasting", "smart_home"]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "environment": settings.environment,
        "version": "3.1.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )