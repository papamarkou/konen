from dataclasses import dataclass, field
from io import StringIO
from typing import Any

from rich.console import Console

from konen import EngineFactory, LLMEngine, LLMResponse, Workspace
from konen.cli import app as app_module
from konen.cli.renderer import CLIRenderer
from konen.cli.session import CLISession


@dataclass(slots=True)
class _FakeEngine(LLMEngine):
    calls: list[dict[str, Any]] = field(default_factory=list, init=False)

    def _initialize_client(self) -> object:
        return object()

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
                "provider_state": provider_state,
            }
        )
        return LLMResponse(
            text=(
                "The posterior probability remains above the decision threshold "
                "under the wider prior."
            ),
            metadata={
                **self._build_response_metadata(),
                "token_usage": {
                    "input_tokens": 180,
                    "output_tokens": 18,
                },
            },
        )


class _ScriptedPromptSession:
    def __init__(self, actions: list[str]) -> None:
        self.actions = list(actions)
        self.calls: list[tuple[str, object | None]] = []

    def prompt(self, label: str, completer: object | None = None) -> str:
        self.calls.append((label, completer))
        if not self.actions:
            raise AssertionError("No scripted CLI input remains.")
        return self.actions.pop(0)


def test_cli_e2e_no_network(tmp_path, monkeypatch) -> None:
    workspace = Workspace(tmp_path / "bayesian-decision-study")
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
    workspace.add_engine(
        engine_uid="gpt-analysis",
        model_family="gpt",
        model_name="gpt-5.6-luna",
    )
    workspace.add_engine(
        engine_uid="gemini-analysis",
        model_family="gemini",
        model_name="gemini-3.5-flash-lite",
    )
    workspace.assign_engine("bayesian-analyst", "gpt-analysis")
    workspace.assign_engine("decision-analyst", "gemini-analysis")
    workspace.append_to_memory(
        "prior_notes.md",
        "Baseline prior probability is 0.40.\n",
    )
    workspace.append_to_scratchpad(
        "bayesian-analyst",
        "Check sensitivity to wider priors before concluding.\n",
    )

    built_engines: list[_FakeEngine] = []

    def build_fake_engine(
        model_family: str,
        model_name: str,
        config: dict[str, Any] | None = None,
        uid: str | None = None,
    ) -> _FakeEngine:
        engine = _FakeEngine(
            model_family=model_family,
            model_name=model_name,
            config={} if config is None else config,
            uid="fake-analysis" if uid is None else uid,
        )
        built_engines.append(engine)
        return engine

    monkeypatch.setattr(
        EngineFactory,
        "build",
        staticmethod(build_fake_engine),
    )

    prompt = "Reassess the posterior probability under the wider prior."
    scripted_session = _ScriptedPromptSession(
        [
            "/agent select bayesian-analyst",
            prompt,
            "/history",
            "/state",
            "/quit",
        ]
    )
    monkeypatch.setattr(
        app_module,
        "PromptSession",
        lambda **kwargs: scripted_session,
    )

    cli = CLISession.from_path(workspace.root_path)

    buffer = StringIO()
    cli.renderer = CLIRenderer(
        Console(
            file=buffer,
            width=160,
            color_system=None,
            force_terminal=False,
        )
    )

    cli.run()

    assert cli.orchestrator.agent_uid == "bayesian-analyst"
    assert sorted(engine.uid for engine in built_engines) == [
        "gemini-analysis",
        "gpt-analysis",
    ]

    gpt_engine = next(
        engine for engine in built_engines if engine.uid == "gpt-analysis"
    )
    gemini_engine = next(
        engine for engine in built_engines if engine.uid == "gemini-analysis"
    )

    assert len(gpt_engine.calls) == 1
    assert gemini_engine.calls == []

    call = gpt_engine.calls[0]
    assert call["provider_state"] is None
    assert prompt in call["prompt"]
    assert "# User prompt" in call["prompt"]
    assert "Baseline prior probability is 0.40." in call["system_instruction"]
    assert (
        "Check sensitivity to wider priors before concluding."
        in call["system_instruction"]
    )
    assert "Bayesian analyst" in call["system_instruction"]
    assert "Decision analyst" in call["system_instruction"]

    state = cli.orchestrator.interaction_state
    assert len(state.transcript_history.rounds) == 1
    assert len(state.provider_history.rounds) == 0

    transcript = state.transcript_history.last_round
    assert transcript is not None
    assert transcript.agent_uid == "bayesian-analyst"
    assert transcript.user_prompt == prompt
    assert transcript.agent_output == (
        "The posterior probability remains above the decision threshold "
        "under the wider prior."
    )
    assert transcript.metadata == {
        "history_limit": None,
        "engine_uid": "gpt-analysis",
        "model_family": "gpt",
        "model_name": "gpt-5.6-luna",
        "token_usage": {
            "input_tokens": 180,
            "output_tokens": 18,
        },
    }

    output = buffer.getvalue()
    assert "Konen" in output
    assert "No active agent. Use /agent list and /agent select <uid>." in output
    assert "Active agent" in output
    assert "Bayesian analyst (bayesian-analyst)" in output
    assert "gpt-analysis" in output
    assert "gpt / gpt-5.6-luna" in output
    assert "posterior probability remains above the decision threshold" in output
    assert "Transcript history" in output
    assert prompt in output
    assert "Interaction state" in output
    assert "Transcript rounds" in output
    assert "Provider rounds" in output
    assert "completed" in output

    assert [label for label, _ in scripted_session.calls] == [
        "konen> ",
        "konen[bayesian-analyst]> ",
        "konen[bayesian-analyst]> ",
        "konen[bayesian-analyst]> ",
        "konen[bayesian-analyst]> ",
    ]
