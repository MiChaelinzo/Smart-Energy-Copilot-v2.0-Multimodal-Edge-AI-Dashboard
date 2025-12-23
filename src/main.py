"""Smart Energy Copilot v2.0 - Main application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config.settings import settings
from src.config.logging import setup_logging, get_logger
from src.components.ocr_api import router as ocr_router
from src.components.ai_api import router as ai_router

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Smart Energy Copilot v2.0",
    description="AI-powered energy optimization dashboard for edge deployment",
    version="2.0.0",
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


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("Starting Smart Energy Copilot v2.0", 
                environment=settings.environment,
                debug=settings.debug)
    
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


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Smart Energy Copilot v2.0",
        "version": "2.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "environment": settings.environment,
        "version": "2.0.0"
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