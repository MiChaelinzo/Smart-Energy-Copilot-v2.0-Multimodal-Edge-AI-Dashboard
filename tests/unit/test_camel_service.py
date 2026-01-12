"""Unit tests for the Camel-AI playbook helper."""

from src.services.camel_service import CamelEnergyPlaybook


def test_generate_playbook_with_minimal_metrics():
    service = CamelEnergyPlaybook()
    payload = service.generate_playbook(
        goal="reduce peak demand",
        recent_metrics={"peak_kwh": 140.0, "base_kwh": 90.0},
        context="lab sensors running overnight",
    )

    assert payload["framework"] == "camel-ai"
    assert len(payload["conversation"]) == 3
    assert any("peak" in str(item["area"]).lower() for item in payload["focus_areas"])


def test_generate_playbook_without_metrics_defaults_to_baseline():
    service = CamelEnergyPlaybook()
    payload = service.generate_playbook(goal="baseline")

    assert payload["framework"] == "camel-ai"
    assert payload["focus_areas"]
    assert len(payload["conversation"]) == 3
