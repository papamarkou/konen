import pytest

from konen import ProviderHistory, ProviderRound


def test_provider_history_defaults() -> None:
    history = ProviderHistory()

    assert history.rounds == []
    assert history.last_round is None


def test_provider_history_append() -> None:
    first = ProviderRound(
        transcript_uid="550e8400-e29b-41d4-a716-446655440000",
        agent_uid="bayesian-analyst",
        engine_uid="gpt-analysis",
        model_family="gpt",
        model_name="gpt-5.6-luna",
        history_start_uid="550e8400-e29b-41d4-a716-446655440000",
        state={"response_id": "resp_01HXYZ123"},
    )
    second = ProviderRound(
        transcript_uid="7c9e6679-7425-40de-944b-e07fc1f90ae7",
        agent_uid="bayesian-analyst",
        engine_uid="gpt-analysis",
        model_family="gpt",
        model_name="gpt-5.6-luna",
        history_start_uid="550e8400-e29b-41d4-a716-446655440000",
        state={"response_id": "resp_01HXYZ456"},
    )
    history = ProviderHistory()

    history.append(first)
    history.append(second)

    assert history.rounds == [first, second]
    assert history.last_round is second


def test_provider_history_duplicate_transcript() -> None:
    first = ProviderRound(
        transcript_uid="550e8400-e29b-41d4-a716-446655440000",
        agent_uid="bayesian-analyst",
        engine_uid="gpt-analysis",
        model_family="gpt",
        model_name="gpt-5.6-luna",
        history_start_uid="550e8400-e29b-41d4-a716-446655440000",
        state={"response_id": "resp_01HXYZ123"},
    )
    duplicate = ProviderRound(
        transcript_uid="550e8400-e29b-41d4-a716-446655440000",
        agent_uid="bayesian-analyst",
        engine_uid="gpt-analysis",
        model_family="gpt",
        model_name="gpt-5.6-luna",
        history_start_uid="550e8400-e29b-41d4-a716-446655440000",
        state={"response_id": "resp_01HXYZ456"},
    )
    history = ProviderHistory(rounds=[first])

    with pytest.raises(ValueError, match="Provider state already exists"):
        history.append(duplicate)


def test_provider_history_clear() -> None:
    round = ProviderRound(
        transcript_uid="550e8400-e29b-41d4-a716-446655440000",
        agent_uid="bayesian-analyst",
        engine_uid="gpt-analysis",
        model_family="gpt",
        model_name="gpt-5.6-luna",
        history_start_uid="550e8400-e29b-41d4-a716-446655440000",
        state={"response_id": "resp_01HXYZ123"},
    )
    history = ProviderHistory(rounds=[round])

    history.clear()

    assert history.rounds == []
    assert history.last_round is None
