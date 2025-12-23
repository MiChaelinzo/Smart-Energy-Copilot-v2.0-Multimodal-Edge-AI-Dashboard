"""Application configuration settings."""

import os
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    model_config = ConfigDict(env_file=".env", case_sensitive=False)
    
    # Environment
    environment: str = Field(default="development", validation_alias="ENVIRONMENT")
    debug: bool = Field(default=False, validation_alias="DEBUG")
    
    # Database
    database_url: str = Field(default="sqlite:///data/energy_copilot.db", validation_alias="DATABASE_URL")
    
    # Logging
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, validation_alias="LOG_FILE")
    
    # API
    api_host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    api_port: int = Field(default=8000, validation_alias="API_PORT")
    
    # AI Model
    model_path: str = Field(default="models/", validation_alias="MODEL_PATH")
    model_cache_size: int = Field(default=1024, validation_alias="MODEL_CACHE_SIZE")  # MB
    
    # ERNIE Model
    ernie_model_path: str = Field(default="models/ernie-energy", validation_alias="ERNIE_MODEL_PATH")
    ernie_max_sequence_length: int = Field(default=512, validation_alias="ERNIE_MAX_SEQUENCE_LENGTH")
    ernie_batch_size: int = Field(default=8, validation_alias="ERNIE_BATCH_SIZE")
    ernie_inference_timeout: float = Field(default=1.0, validation_alias="ERNIE_INFERENCE_TIMEOUT")  # seconds
    
    # OCR
    ocr_confidence_threshold: float = Field(default=0.7, validation_alias="OCR_CONFIDENCE_THRESHOLD")
    ocr_max_file_size: int = Field(default=10, validation_alias="OCR_MAX_FILE_SIZE")  # MB
    
    # IoT
    mqtt_broker_host: Optional[str] = Field(default=None, validation_alias="MQTT_BROKER_HOST")
    mqtt_broker_port: int = Field(default=1883, validation_alias="MQTT_BROKER_PORT")
    iot_data_retention_days: int = Field(default=365, validation_alias="IOT_DATA_RETENTION_DAYS")
    
    # Security
    secret_key: str = Field(default="dev-secret-key", validation_alias="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, validation_alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Edge deployment
    max_memory_usage_mb: int = Field(default=1800, validation_alias="MAX_MEMORY_USAGE_MB")  # Leave 200MB buffer
    enable_model_quantization: bool = Field(default=True, validation_alias="ENABLE_MODEL_QUANTIZATION")


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings