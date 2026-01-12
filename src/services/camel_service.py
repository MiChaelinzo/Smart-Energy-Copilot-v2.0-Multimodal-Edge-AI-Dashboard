"""Lightweight Camel-AI orchestration helpers for the dashboard."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional
import logging

from src.config.settings import get_settings

try:
    from camel.messages import BaseMessage

    _camel_available = True
except ModuleNotFoundError:
    BaseMessage = None  # type: ignore
    _camel_available = False

logger = logging.getLogger(__name__)


class CamelEnergyPlaybook:
    """Compose structured recommendations using Camel-AI message schema."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.camel_available = _camel_available and self.settings.camel_enabled

    def generate_playbook(
        self,
        goal: str,
        recent_metrics: Optional[Dict[str, float]] = None,
        context: Optional[str] = None,
    ) -> Dict[str, object]:
        """Return a small, deterministic playbook payload."""

        metrics = recent_metrics or {}
        focus_areas = self._derive_focus_areas(metrics)
        summary = self._build_summary(goal, focus_areas, context)
        conversation = self._build_conversation(goal, summary, focus_areas)

        return {
            "framework": "camel-ai",
            "camel_available": self.camel_available,
            "summary": summary,
            "focus_areas": focus_areas,
            "conversation": conversation,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _derive_focus_areas(self, metrics: Dict[str, float]) -> List[Dict[str, object]]:
        focus: List[Dict[str, object]] = []

        if not metrics:
            return [
                {"area": "baseline", "action": "Collect a 7-day sample of load and sensor data"},
                {"area": "safety", "action": "Verify alerting webhooks and edge watchdogs"},
            ]

        peak = metrics.get("peak_kwh")
        base = metrics.get("base_kwh")
        if peak and base and peak > base * 1.3:
            focus.append(
                {
                    "area": "peak shaving",
                    "action": "Schedule heavy loads away from peak windows and pre-cool thermal assets",
                    "confidence": 0.78,
                }
            )

        power_factor = metrics.get("power_factor")
        if power_factor and power_factor < 0.92:
            focus.append(
                {
                    "area": "power factor",
                    "action": "Enable capacitor bank automation to keep pf above 0.95",
                    "confidence": 0.7,
                }
            )

        co2_intensity = metrics.get("co2_intensity")
        if co2_intensity and co2_intensity > 400:
            focus.append(
                {
                    "area": "decarbonization",
                    "action": "Shift discretionary workloads to cleaner grid windows",
                    "confidence": 0.74,
                }
            )

        if not focus:
            focus.append(
                {
                    "area": "efficiency tuning",
                    "action": "Audit idle loads and tighten device-level sleep thresholds",
                    "confidence": 0.68,
                }
            )

        return focus

    def _build_summary(self, goal: str, focus: List[Dict[str, object]], context: Optional[str]) -> str:
        bullets = [f"- {item['area']}: {item['action']}" for item in focus]
        ctx = f"\nContext: {context}" if context else ""
        return (
            f"Goal: {goal or self.settings.camel_goal}\n"
            f"Priority actions:\n" + "\n".join(bullets) + ctx
        )

    def _build_conversation(
        self, goal: str, summary: str, focus: List[Dict[str, object]]
    ) -> List[Dict[str, object]]:
        base_conversation = []

        system_content = (
            "You are a CAMEL-style multi-agent energy copilot focused on safe, "
            "edge-optimized orchestration."
        )
        user_content = (
            f"Our objective is: {goal or self.settings.camel_goal}. "
            f"Here is the latest operating snapshot:\n{summary}"
        )
        assistant_content = "Recommended next steps:\n" + "\n".join(
            f"* {item['area']}: {item['action']}" for item in focus
        )

        if self.camel_available and BaseMessage:
            system_msg = BaseMessage.make_system_message("EnergyDirector", system_content)
            user_msg = BaseMessage.make_user_message("OpsManager", user_content)
            assistant_msg = BaseMessage.make_assistant_message("CamelOrchestrator", assistant_content)
            base_conversation = [
                system_msg.to_dict(),
                user_msg.to_dict(),
                assistant_msg.to_dict(),
            ]
        else:
            base_conversation = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]

        return base_conversation
