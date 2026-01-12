"""Camel-AI helper endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Dict, Optional

from src.services.camel_service import CamelEnergyPlaybook

router = APIRouter(prefix="/api/v1/camel", tags=["Camel-AI"])


class CamelPlaybookRequest(BaseModel):
    goal: str = Field(..., description="High-level energy intent or question")
    metrics: Optional[Dict[str, float]] = Field(
        default=None, description="Recent KPIs to anchor the plan"
    )
    context: Optional[str] = Field(default=None, description="Optional narrative context")


@router.post("/playbook")
async def generate_camel_playbook(request: CamelPlaybookRequest):
    """Generate a light, deterministic playbook using Camel-style messages."""
    service = CamelEnergyPlaybook()
    return service.generate_playbook(request.goal, request.metrics, request.context)
