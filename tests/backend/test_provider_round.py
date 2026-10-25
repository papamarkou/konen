from konen import ProviderRound


def test_provider_round_preserves_fields() -> None:
    round = ProviderRound(
        transcript_uid="7c9e6679-7425-40de-944b-e07fc1f90ae7",
        agent_uid="bayesian-analyst",
        engine_uid="gpt-analysis",
        model_family="gpt",
        model_name="gpt-5.6-luna",
        history_start_uid="550e8400-e29b-41d4-a716-446655440000",
        state={"response_id": "resp_01HXYZ123"},
    )

    assert round.transcript_uid == "7c9e6679-7425-40de-944b-e07fc1f90ae7"
    assert round.agent_uid == "bayesian-analyst"
    assert round.engine_uid == "gpt-analysis"
    assert round.model_family == "gpt"
    assert round.model_name == "gpt-5.6-luna"
    assert round.history_start_uid == "550e8400-e29b-41d4-a716-446655440000"
    assert round.state == {"response_id": "resp_01HXYZ123"}


def test_provider_round_deep_copies_state_input() -> None:
    state = {"response": {"id": "resp_01HXYZ123", "status": "completed"}}

    round = ProviderRound(
        transcript_uid="7c9e6679-7425-40de-944b-e07fc1f90ae7",
        agent_uid="bayesian-analyst",
        engine_uid="gpt-analysis",
        model_family="gpt",
        model_name="gpt-5.6-luna",
        history_start_uid="550e8400-e29b-41d4-a716-446655440000",
        state=state,
    )

    state["response"]["status"] = "changed"

    assert round.state == {"response": {"id": "resp_01HXYZ123", "status": "completed"}}


def test_provider_round_to_dict_structure() -> None:
    round = ProviderRound(
        transcript_uid="7c9e6679-7425-40de-944b-e07fc1f90ae7",
        agent_uid="bayesian-analyst",
        engine_uid="gpt-analysis",
        model_family="gpt",
        model_name="gpt-5.6-luna",
        history_start_uid="550e8400-e29b-41d4-a716-446655440000",
        state={"response_id": "resp_01HXYZ123"},
    )

    assert round.to_dict() == {
        "transcript_uid": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
        "agent_uid": "bayesian-analyst",
        "engine_uid": "gpt-analysis",
        "model_family": "gpt",
        "model_name": "gpt-5.6-luna",
        "history_start_uid": "550e8400-e29b-41d4-a716-446655440000",
        "state": {"response_id": "resp_01HXYZ123"},
    }


def test_provider_round_to_dict_deep_copies_state() -> None:
    round = ProviderRound(
        transcript_uid="7c9e6679-7425-40de-944b-e07fc1f90ae7",
        agent_uid="bayesian-analyst",
        engine_uid="gpt-analysis",
        model_family="gpt",
        model_name="gpt-5.6-luna",
        history_start_uid="550e8400-e29b-41d4-a716-446655440000",
        state={"response": {"id": "resp_01HXYZ123", "status": "completed"}},
    )

    data = round.to_dict()
    data["state"]["response"]["status"] = "changed"

    assert round.state["response"]["status"] == "completed"
