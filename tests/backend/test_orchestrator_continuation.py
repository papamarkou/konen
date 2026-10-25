import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from konen import (
    AITeam,
    EngineRouter,
    LLMEngine,
    LLMPool,
    LLMResponse,
    Orchestrator,
    ProviderHistory,
    Workspace,
)


@dataclass(slots=True)
class _StatefulEngine(LLMEngine):
    calls: list[dict[str, Any]] = field(default_factory=list, init=False)
    prepared_states: list[list[dict[str, Any]]] = field(
        default_factory=list,
        init=False,
    )
    responses: list[LLMResponse] = field(default_factory=list, init=False)

    def _initialize_client(self) -> object:
        return object()

    def prepare_provider_state(
        self,
        states: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        copied_states = copy.deepcopy(states)
        self.prepared_states.append(copied_states)
        return {"chain": copied_states}

    def generate_response(
        self,
        prompt: str,
        system_instruction: str | None = None,
        provider_state: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "prompt": prompt,
                "system_instruction": system_instruction,
                "provider_state": copy.deepcopy(provider_state),
            }
        )
        turn = len(self.calls)

        if self.uid == "gemini-analysis":
            outputs = [
                "Action A has the higher expected utility.",
                "Action A remains preferred after sensitivity analysis.",
                "The decision is stable across the tested scenarios.",
            ]
        else:
            outputs = [
                "The baseline posterior probability is 0.82.",
                "The wider prior lowers the posterior probability to 0.79.",
                "The posterior remains above the 0.75 decision threshold.",
            ]

        metadata = self._build_response_metadata()
        metadata["token_usage"] = {
            "input_token_count": 100 + turn,
            "output_token_count": 20 + turn,
            "total_token_count": 120 + 2 * turn,
        }
        response = LLMResponse(
            text=outputs[min(turn - 1, len(outputs) - 1)],
            metadata=metadata,
            provider_state={
                "turn": turn,
                "response_id": f"{self.uid}-state-{turn}",
            },
        )
        self.responses.append(response)
        return response


def _workspace(tmp_path: Path) -> Workspace:
    workspace = Workspace(tmp_path / "continuation-study")
    workspace.initialize()
    workspace.add_agent(
        agent_uid="bayesian-analyst",
        name="Bayesian analyst",
        description="Evaluates uncertainty using Bayesian inference.",
    )
    workspace.add_agent(
        agent_uid="decision-analyst",
        name="Decision analyst",
        description=(
            "Compares actions using expected utility and decision-theoretic trade-offs."
        ),
    )
    workspace.append_to_memory(
        "research/prior_notes.md",
        "The baseline prior odds favor the first hypothesis by 2:1.\n",
    )
    workspace.append_to_scratchpad(
        "bayesian-analyst",
        "Compare prior scales of 1 and 2 before reporting the conclusion.\n",
    )
    workspace.append_to_scratchpad(
        "decision-analyst",
        "Compare expected utility under baseline and sensitivity scenarios.\n",
    )
    return workspace


def _orchestrator(
    tmp_path: Path,
) -> tuple[Orchestrator, _StatefulEngine, _StatefulEngine]:
    workspace = _workspace(tmp_path)
    gpt = _StatefulEngine(
        model_family="gpt",
        model_name="gpt-5.6-luna",
        uid="gpt-analysis",
    )
    gemini = _StatefulEngine(
        model_family="gemini",
        model_name="gemini-3.5-flash-lite",
        uid="gemini-analysis",
    )
    orchestrator = Orchestrator(
        workspace=workspace,
        ai_team=AITeam.from_workspace(workspace),
        llm_pool=LLMPool([gpt, gemini]),
        engine_router=EngineRouter(
            {
                "bayesian-analyst": "gpt-analysis",
                "decision-analyst": "gemini-analysis",
            }
        ),
        agent_uid="bayesian-analyst",
    )
    return orchestrator, gpt, gemini


def test_orchestrator_response_metadata(tmp_path: Path) -> None:
    orchestrator, gpt, _ = _orchestrator(tmp_path)

    orchestrator.run_turn(
        "Estimate the posterior probability under the baseline prior.",
        history_limit=2,
    )

    transcript = orchestrator.interaction_state.transcript_history.last_round
    assert transcript is not None
    assert transcript.metadata == {
        "history_limit": 2,
        "engine_uid": "gpt-analysis",
        "model_family": "gpt",
        "model_name": "gpt-5.6-luna",
        "token_usage": {
            "input_token_count": 101,
            "output_token_count": 21,
            "total_token_count": 122,
        },
    }

    gpt.responses[0].metadata["token_usage"]["input_token_count"] = 999
    assert transcript.metadata["token_usage"]["input_token_count"] == 101


def test_orchestrator_provider_state(tmp_path: Path) -> None:
    orchestrator, gpt, _ = _orchestrator(tmp_path)

    orchestrator.run_turn(
        "Estimate the posterior probability under the baseline prior."
    )

    transcript = orchestrator.interaction_state.transcript_history.last_round
    provider_round = orchestrator.interaction_state.provider_history.last_round
    assert transcript is not None
    assert provider_round is not None
    assert provider_round.transcript_uid == transcript.uid
    assert provider_round.agent_uid == "bayesian-analyst"
    assert provider_round.engine_uid == "gpt-analysis"
    assert provider_round.model_family == "gpt"
    assert provider_round.model_name == "gpt-5.6-luna"
    assert provider_round.history_start_uid == transcript.uid
    assert provider_round.state == {
        "turn": 1,
        "response_id": "gpt-analysis-state-1",
    }

    gpt.responses[0].provider_state["response_id"] = "changed"
    assert provider_round.state["response_id"] == "gpt-analysis-state-1"


def test_orchestrator_provider_continuation(tmp_path: Path) -> None:
    orchestrator, gpt, _ = _orchestrator(tmp_path)
    first_prompt = "Estimate the posterior probability under the baseline prior."
    second_prompt = "Reassess the posterior probability under the wider prior."
    third_prompt = "Check whether the posterior remains above the threshold."

    orchestrator.run_turn(first_prompt)
    first_transcript = orchestrator.interaction_state.transcript_history.last_round
    assert first_transcript is not None

    orchestrator.run_turn(second_prompt)

    assert gpt.prepared_states == [
        [
            {
                "turn": 1,
                "response_id": "gpt-analysis-state-1",
            }
        ]
    ]
    assert gpt.calls[1]["provider_state"] == {
        "chain": [
            {
                "turn": 1,
                "response_id": "gpt-analysis-state-1",
            }
        ]
    }
    assert "# Transcript history" not in gpt.calls[1]["prompt"]
    assert first_prompt not in gpt.calls[1]["prompt"]
    assert "The baseline posterior probability is 0.82." not in (gpt.calls[1]["prompt"])
    assert second_prompt in gpt.calls[1]["prompt"]

    orchestrator.run_turn(third_prompt)

    assert gpt.prepared_states[1] == [
        {
            "turn": 1,
            "response_id": "gpt-analysis-state-1",
        },
        {
            "turn": 2,
            "response_id": "gpt-analysis-state-2",
        },
    ]
    assert gpt.calls[2]["provider_state"] == {"chain": gpt.prepared_states[1]}
    provider_round = orchestrator.interaction_state.provider_history.last_round
    assert provider_round is not None
    assert provider_round.history_start_uid == first_transcript.uid


def test_orchestrator_continuation_resets_for_history_window(
    tmp_path: Path,
) -> None:
    orchestrator, gpt, _ = _orchestrator(tmp_path)
    first_prompt = "Estimate the posterior probability under the baseline prior."
    second_prompt = "Reassess the posterior probability under the wider prior."
    third_prompt = "Check whether the posterior remains above the threshold."

    orchestrator.run_turn(first_prompt)
    orchestrator.run_turn(second_prompt)
    second_transcript = orchestrator.interaction_state.transcript_history.last_round
    assert second_transcript is not None

    orchestrator.run_turn(third_prompt, history_limit=1)

    assert len(gpt.prepared_states) == 1
    assert gpt.calls[2]["provider_state"] is None
    assert first_prompt not in gpt.calls[2]["prompt"]
    assert "The baseline posterior probability is 0.82." not in (gpt.calls[2]["prompt"])
    assert second_prompt in gpt.calls[2]["prompt"]
    assert (
        "The wider prior lowers the posterior probability to 0.79."
        in (gpt.calls[2]["prompt"])
    )
    assert third_prompt in gpt.calls[2]["prompt"]

    provider_round = orchestrator.interaction_state.provider_history.last_round
    assert provider_round is not None
    assert provider_round.history_start_uid == second_transcript.uid


def test_orchestrator_continuation_resets_for_agent(tmp_path: Path) -> None:
    orchestrator, gpt, gemini = _orchestrator(tmp_path)
    first_prompt = "Estimate the posterior probability under the baseline prior."

    orchestrator.run_turn(first_prompt)
    orchestrator.agent_uid = "decision-analyst"
    orchestrator.run_turn("Compare the expected utility of the available actions.")

    assert len(gpt.calls) == 1
    assert gemini.calls[0]["provider_state"] is None
    assert gemini.prepared_states == []
    assert first_prompt in gemini.calls[0]["prompt"]
    assert "The baseline posterior probability is 0.82." in (gemini.calls[0]["prompt"])


def test_orchestrator_continuation_resets_for_engine(tmp_path: Path) -> None:
    orchestrator, gpt, gemini = _orchestrator(tmp_path)
    first_prompt = "Estimate the posterior probability under the baseline prior."

    orchestrator.run_turn(first_prompt)
    orchestrator.engine_router.assign_engine(
        "bayesian-analyst",
        "gemini-analysis",
    )
    orchestrator.run_turn("Reassess the posterior using the alternate engine.")

    assert len(gpt.calls) == 1
    assert gemini.calls[0]["provider_state"] is None
    assert gemini.prepared_states == []
    assert first_prompt in gemini.calls[0]["prompt"]


def test_orchestrator_continuation_resets_for_model(tmp_path: Path) -> None:
    orchestrator, gpt, _ = _orchestrator(tmp_path)
    first_prompt = "Estimate the posterior probability under the baseline prior."

    orchestrator.run_turn(first_prompt)
    orchestrator.llm_pool.remove_engine("gpt-analysis")
    revised = _StatefulEngine(
        model_family="gpt",
        model_name="gpt-5.6-terra",
        uid="gpt-analysis",
    )
    orchestrator.llm_pool.add_engine(revised)

    orchestrator.run_turn("Reassess the posterior using the revised model.")

    assert len(gpt.calls) == 1
    assert revised.calls[0]["provider_state"] is None
    assert revised.prepared_states == []
    assert first_prompt in revised.calls[0]["prompt"]
    provider_round = orchestrator.interaction_state.provider_history.last_round
    assert provider_round is not None
    assert provider_round.model_name == "gpt-5.6-terra"


def test_orchestrator_inconsistent_provider_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, _, _ = _orchestrator(tmp_path)
    orchestrator.run_turn(
        "Estimate the posterior probability under the baseline prior."
    )

    def inconsistent_chain(
        self: ProviderHistory,
        agent_uid: str,
        engine_uid: str,
        model_family: str,
        model_name: str,
        history_start_uid: str,
    ) -> list[Any]:
        return []

    monkeypatch.setattr(
        ProviderHistory,
        "for_chain",
        inconsistent_chain,
    )

    with pytest.raises(
        RuntimeError, match="Provider continuation chain is inconsistent"
    ):
        orchestrator.run_turn(
            "Reassess the posterior probability under the wider prior."
        )
