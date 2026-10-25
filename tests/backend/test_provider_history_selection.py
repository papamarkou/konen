import json
from pathlib import Path

from konen import ProviderHistory, ProviderRound


def _provider_history() -> ProviderHistory:
    return ProviderHistory(
        rounds=[
            ProviderRound(
                transcript_uid="550e8400-e29b-41d4-a716-446655440000",
                agent_uid="bayesian-analyst",
                engine_uid="gpt-analysis",
                model_family="gpt",
                model_name="gpt-5.6-luna",
                history_start_uid="550e8400-e29b-41d4-a716-446655440000",
                state={"response": {"id": "resp_01HXYZ123", "status": "completed"}},
            ),
            ProviderRound(
                transcript_uid="7c9e6679-7425-40de-944b-e07fc1f90ae7",
                agent_uid="bayesian-analyst",
                engine_uid="gpt-analysis",
                model_family="gpt",
                model_name="gpt-5.6-luna",
                history_start_uid="550e8400-e29b-41d4-a716-446655440000",
                state={"response": {"id": "resp_01HXYZ456", "status": "completed"}},
            ),
            ProviderRound(
                transcript_uid="123e4567-e89b-12d3-a456-426614174000",
                agent_uid="decision-analyst",
                engine_uid="gpt-analysis",
                model_family="gpt",
                model_name="gpt-5.6-luna",
                history_start_uid="123e4567-e89b-12d3-a456-426614174000",
                state={"response_id": "resp_01HXYZ789"},
            ),
            ProviderRound(
                transcript_uid="9b2d7c4e-3f18-4a6b-8c21-5d7e9f103456",
                agent_uid="bayesian-analyst",
                engine_uid="gemini-analysis",
                model_family="gemini",
                model_name="gemini-3.5-flash-lite",
                history_start_uid="9b2d7c4e-3f18-4a6b-8c21-5d7e9f103456",
                state={"response_id": "gemini-response-42"},
            ),
        ]
    )


def test_provider_history_for_chain() -> None:
    history = _provider_history()

    selected = history.for_chain(
        agent_uid="bayesian-analyst",
        engine_uid="gpt-analysis",
        model_family="gpt",
        model_name="gpt-5.6-luna",
        history_start_uid="550e8400-e29b-41d4-a716-446655440000",
    )

    assert [round.transcript_uid for round in selected] == [
        "550e8400-e29b-41d4-a716-446655440000",
        "7c9e6679-7425-40de-944b-e07fc1f90ae7",
    ]
    selected[0].state["response"]["status"] = "changed"
    assert history.rounds[0].state["response"]["status"] == "completed"


def test_provider_history_for_chain_empty() -> None:
    history = _provider_history()

    selected = history.for_chain(
        agent_uid="bayesian-analyst",
        engine_uid="gpt-analysis",
        model_family="gpt",
        model_name="gpt-5.6-luna",
        history_start_uid="123e4567-e89b-12d3-a456-426614174000",
    )

    assert selected == []


def test_provider_history_last_for() -> None:
    history = _provider_history()

    selected = history.last_for(
        agent_uid="bayesian-analyst",
        engine_uid="gpt-analysis",
        model_family="gpt",
        model_name="gpt-5.6-luna",
    )

    assert selected is history.rounds[1]


def test_provider_history_last_for_missing() -> None:
    history = _provider_history()

    selected = history.last_for(
        agent_uid="causal-analyst",
        engine_uid="gpt-analysis",
        model_family="gpt",
        model_name="gpt-5.6-luna",
    )

    assert selected is None


def test_provider_history_clear_for() -> None:
    history = _provider_history()

    history.clear_for(agent_uid="bayesian-analyst", engine_uid="gpt-analysis")

    assert [round.transcript_uid for round in history.rounds] == [
        "123e4567-e89b-12d3-a456-426614174000",
        "9b2d7c4e-3f18-4a6b-8c21-5d7e9f103456",
    ]


def test_provider_history_to_dict() -> None:
    history = ProviderHistory(
        rounds=[
            ProviderRound(
                transcript_uid="550e8400-e29b-41d4-a716-446655440000",
                agent_uid="bayesian-analyst",
                engine_uid="gpt-analysis",
                model_family="gpt",
                model_name="gpt-5.6-luna",
                history_start_uid="550e8400-e29b-41d4-a716-446655440000",
                state={"response_id": "resp_01HXYZ123"},
            )
        ]
    )

    assert history.to_dict() == [
        {
            "transcript_uid": "550e8400-e29b-41d4-a716-446655440000",
            "agent_uid": "bayesian-analyst",
            "engine_uid": "gpt-analysis",
            "model_family": "gpt",
            "model_name": "gpt-5.6-luna",
            "history_start_uid": "550e8400-e29b-41d4-a716-446655440000",
            "state": {"response_id": "resp_01HXYZ123"},
        }
    ]


def test_provider_history_save_json(tmp_path: Path) -> None:
    history = _provider_history()
    path = tmp_path / "exports" / "provider_history.json"

    history.save_json(path)

    text = path.read_text(encoding="utf-8")
    assert json.loads(text) == history.to_dict()
    assert text.endswith("\n")
