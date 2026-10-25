from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

from rich.console import Console

from konen import (
    AIAgent,
    AITeam,
    EngineRouter,
    InteractionState,
    LLMEngine,
    LLMPool,
    LLMResponse,
    Orchestrator,
    ProviderHistory,
    ProviderRound,
    TranscriptRound,
)
from konen.cli.commands import CommandGroup, CommandSpec
from konen.cli.renderer import CLIRenderer


@dataclass(slots=True)
class _FakeEngine(LLMEngine):
    def _initialize_client(self) -> object:
        return object()

    def generate_response(
        self,
        prompt: str,
        system_instruction: str | None = None,
        provider_state: dict[str, Any] | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            text="The posterior probability remains above the decision threshold."
        )


@dataclass
class _Workspace:
    root_path: Path = Path("/research/bayesian-decision-study")
    memory_path: Path = Path("/research/bayesian-decision-study/memory")
    scratchpad_path: Path = Path("/research/bayesian-decision-study/scratchpad")
    config_path: Path = Path("/research/bayesian-decision-study/config")

    def list_workspace_files(self) -> dict[str, list[str]]:
        return {
            "memory": ["prior_notes.md", "literature/robustness.md"],
            "scratchpad": ["bayesian-analyst.md", "decision-analyst.md"],
            "config": ["agents.json", "engines.json", "agent_engine_map.json"],
        }

    def load_agent_engine_map_config(self) -> dict[str, object]:
        return {
            "version": "1.0",
            "agent_engine_map": {
                "bayesian-analyst": "gpt-analysis",
                "decision-analyst": "gemini-analysis",
            },
        }

    def list_agents(self) -> list[str]:
        return ["bayesian-analyst", "decision-analyst"]

    def list_engines(self) -> list[str]:
        return ["gemini-analysis", "gpt-analysis"]


def _renderer() -> tuple[CLIRenderer, StringIO]:
    buffer = StringIO()
    console = Console(
        file=buffer,
        width=160,
        color_system=None,
        force_terminal=False,
    )
    return CLIRenderer(console), buffer


def _output(buffer: StringIO) -> str:
    return buffer.getvalue()


def _team() -> AITeam:
    return AITeam(
        [
            AIAgent(
                name="Bayesian analyst",
                description="Evaluates uncertainty using Bayesian inference.",
                uid="bayesian-analyst",
            ),
            AIAgent(
                name="Decision analyst",
                description=(
                    "Compares actions using expected utility and decision-theoretic trade-offs."
                ),
                uid="decision-analyst",
            ),
        ]
    )


def _pool() -> LLMPool:
    return LLMPool(
        [
            _FakeEngine(
                model_family="gpt",
                model_name="gpt-5.6-luna",
                config={"reasoning_effort": "high"},
                uid="gpt-analysis",
            ),
            _FakeEngine(
                model_family="gemini",
                model_name="gemini-3.5-flash-lite",
                uid="gemini-analysis",
            ),
        ]
    )


def _orchestrator(
    *,
    agent_uid: str | None = "bayesian-analyst",
    router: EngineRouter | None = None,
    state: InteractionState | None = None,
) -> Orchestrator:
    return Orchestrator(
        ai_team=_team(),
        llm_pool=_pool(),
        engine_router=(
            EngineRouter(
                {
                    "bayesian-analyst": "gpt-analysis",
                    "decision-analyst": "gemini-analysis",
                }
            )
            if router is None
            else router
        ),
        agent_uid=agent_uid,
        interaction_state=InteractionState() if state is None else state,
    )


def _provider_history() -> ProviderHistory:
    return ProviderHistory(
        [
            ProviderRound(
                transcript_uid="550e8400-e29b-41d4-a716-446655440000",
                agent_uid="bayesian-analyst",
                engine_uid="gpt-analysis",
                model_family="gpt",
                model_name="gpt-5.6-luna",
                history_start_uid="550e8400-e29b-41d4-a716-446655440000",
                state={"response_id": "resp_bayesian_001"},
            ),
            ProviderRound(
                transcript_uid="7c9e6679-7425-40de-944b-e07fc1f90ae7",
                agent_uid="decision-analyst",
                engine_uid="gemini-analysis",
                model_family="gemini",
                model_name="gemini-3.5-flash-lite",
                history_start_uid="7c9e6679-7425-40de-944b-e07fc1f90ae7",
                state={"interaction_id": "interaction_decision_001"},
            ),
        ]
    )


def test_cli_renderer_banner() -> None:
    renderer, buffer = _renderer()

    renderer.show_banner()

    text = _output(buffer)
    assert "Konen" in text
    assert "Human-centered orchestration of AI agents" in text
    assert "Use /help" in text


def test_cli_renderer_help() -> None:
    renderer, buffer = _renderer()
    groups = (
        CommandGroup("agent", "Agents"),
        CommandGroup("engine", "Engines"),
    )
    commands = (
        CommandSpec(
            "agent",
            "/agent select <uid>",
            "Select the active agent for subsequent prompts.",
            "/agent select",
        ),
        CommandSpec(
            "agent",
            "/agent show <uid>",
            "Show one configured agent.",
            "/agent show",
        ),
        CommandSpec(
            "engine",
            "/engine show <uid>",
            "Show one configured engine.",
            "/engine show",
        ),
    )

    renderer.show_help(groups, commands)

    text = _output(buffer)
    assert "Command groups" in text
    assert "Agents" in text
    assert "/help agent" in text
    assert "2" in text
    assert "Engines" in text
    assert "/help engine" in text
    assert "press Tab for context-aware completion" in text


def test_cli_renderer_help_group() -> None:
    renderer, buffer = _renderer()
    group = CommandGroup("agent", "Agents")
    commands = (
        CommandSpec(
            "agent",
            "/agent select <uid>",
            "Select the active agent for subsequent prompts.",
            "/agent select",
        ),
        CommandSpec(
            "agent",
            "/agent show <uid>",
            "Show one configured agent.",
            "/agent show",
        ),
    )

    renderer.show_help_group(group, commands)

    text = _output(buffer)
    assert "Agents" in text
    assert "/agent select <uid>" in text
    assert "Select the active agent" in text
    assert "/agent show <uid>" in text


def test_cli_renderer_workspace() -> None:
    renderer, buffer = _renderer()
    workspace = _Workspace()
    orchestrator = _orchestrator()

    renderer.show_workspace(workspace, orchestrator)  # type: ignore[arg-type]

    text = _output(buffer)
    assert "Workspace" in text
    assert "/research/bayesian-decision-study" in text
    assert "Agents" in text
    assert "2" in text
    assert "Mappings" in text
    assert "Memory files" in text
    assert "bayesian-analyst" in text


def test_cli_renderer_agents() -> None:
    renderer, buffer = _renderer()

    renderer.show_agents(_orchestrator())

    text = _output(buffer)
    assert "Agents" in text
    assert "* bayesian-analyst" in text
    assert "Bayesian analyst" in text
    assert "Evaluates uncertainty using Bayesian inference." in text
    assert "decision-analyst" in text
    assert "Decision analyst" in text


def test_cli_renderer_agents_empty() -> None:
    renderer, buffer = _renderer()
    orchestrator = Orchestrator(ai_team=AITeam())

    renderer.show_agents(orchestrator)

    assert "No agents are configured." in _output(buffer)


def test_cli_renderer_agents_unmapped() -> None:
    renderer, buffer = _renderer()
    orchestrator = _orchestrator(router=EngineRouter())

    renderer.show_agents(orchestrator)

    text = _output(buffer)
    assert "Bayesian analyst" in text
    assert "Decision analyst" in text


def test_cli_renderer_agent() -> None:
    renderer, buffer = _renderer()

    renderer.show_agent(_orchestrator(), "bayesian-analyst")

    text = _output(buffer)
    assert "Agent · bayesian-analyst" in text
    assert "Bayesian analyst" in text
    assert "Active" in text
    assert "yes" in text
    assert "gpt-analysis" in text
    assert "gpt-5.6-luna" in text


def test_cli_renderer_agent_selected() -> None:
    renderer, buffer = _renderer()

    renderer.show_agent_selected(_orchestrator(), "decision-analyst")

    text = _output(buffer)
    assert "Active agent" in text
    assert "Decision analyst (decision-analyst)" in text
    assert "gemini-analysis" in text
    assert "gemini / gemini-3.5-flash-lite" in text


def test_cli_renderer_engines() -> None:
    renderer, buffer = _renderer()

    renderer.show_engines(_pool())

    text = _output(buffer)
    assert "Engines" in text
    assert "gpt-analysis" in text
    assert "gpt-5.6-luna" in text
    assert "reasoning_effort" in text
    assert "gemini-analysis" in text
    assert "gemini-3.5-flash-lite" in text


def test_cli_renderer_engines_empty() -> None:
    renderer, buffer = _renderer()

    renderer.show_engines(LLMPool())

    assert "No engines are configured." in _output(buffer)


def test_cli_renderer_engine_redacts_secrets() -> None:
    renderer, buffer = _renderer()
    engine = _FakeEngine(
        model_family="gpt",
        model_name="gpt-5.6-luna",
        config={
            "api_key": "sk-secret-value",
            "reasoning_effort": "high",
        },
        uid="gpt-analysis",
    )

    renderer.show_engine(engine)

    text = _output(buffer)
    assert "Engine · gpt-analysis" in text
    assert "gpt-5.6-luna" in text
    assert "reasoning_effort" in text
    assert "high" in text
    assert "[redacted]" in text
    assert "sk-secret-value" not in text


def test_cli_renderer_mappings() -> None:
    renderer, buffer = _renderer()

    renderer.show_mappings(_orchestrator())

    text = _output(buffer)
    assert "Agent-engine mappings" in text
    assert "bayesian-analyst" in text
    assert "Bayesian analyst" in text
    assert "gpt-analysis" in text
    assert "gpt-5.6-luna" in text
    assert "decision-analyst" in text
    assert "gemini-analysis" in text
    assert "gemini-3.5-flash-lite" in text


def test_cli_renderer_mapping() -> None:
    renderer, buffer = _renderer()

    renderer.show_mapping(
        _orchestrator(),
        "bayesian-analyst",
        "gpt-analysis",
    )

    text = _output(buffer)
    assert "Agent-engine mapping" in text
    assert "bayesian-analyst" in text
    assert "Bayesian analyst" in text
    assert "gpt-analysis" in text
    assert "gpt-5.6-luna" in text


def test_cli_renderer_files() -> None:
    renderer, buffer = _renderer()
    files = _Workspace().list_workspace_files()

    renderer.show_files(files, category="memory")

    text = _output(buffer)
    assert "Files · memory" in text
    assert "memory/prior_notes.md" in text
    assert "memory/literature/robustness.md" in text


def test_cli_renderer_files_empty() -> None:
    renderer, buffer = _renderer()

    renderer.show_files(
        {
            "memory": [],
            "scratchpad": ["bayesian-analyst.md"],
            "config": [],
        }
    )

    text = _output(buffer)
    assert "No memory files are present." in text
    assert "scratchpad/bayesian-analyst.md" in text
    assert "No config files are present." in text


def test_cli_renderer_scratchpads() -> None:
    renderer, buffer = _renderer()

    renderer.show_scratchpads(_Workspace())

    text = _output(buffer)
    assert "Files · scratchpad" in text
    assert "Agent" in text
    assert "Path" in text
    assert any(
        "bayesian-analyst" in line and "scratchpad/bayesian-analyst.md" in line
        for line in text.splitlines()
    )
    assert any(
        "decision-analyst" in line and "scratchpad/decision-analyst.md" in line
        for line in text.splitlines()
    )


def test_cli_renderer_file_types() -> None:
    renderer, buffer = _renderer()

    renderer.show_file(
        "memory/posterior_summary.md",
        "# Posterior summary\n\nThe posterior probability is 0.82.\n",
    )
    renderer.show_file(
        "config/analysis.json",
        '{"decision_threshold": 0.75}',
    )
    renderer.show_file(
        "memory/notes.txt",
        "Sensitivity analysis remains stable.",
    )
    renderer.show_file("memory/empty.md", "")

    text = _output(buffer)
    assert "memory/posterior_summary.md" in text
    assert "Posterior summary" in text
    assert "posterior probability is 0.82" in text
    assert "config/analysis.json" in text
    assert "decision_threshold" in text
    assert "memory/notes.txt" in text
    assert "Sensitivity analysis remains stable." in text
    assert "[empty file]" in text


def test_cli_renderer_response() -> None:
    renderer, buffer = _renderer()

    renderer.show_response(
        _orchestrator(),
        "**Posterior conclusion:** the probability remains above 0.75.",
    )

    text = _output(buffer)
    assert "Bayesian analyst · bayesian-analyst" in text
    assert "Posterior conclusion" in text
    assert "probability remains above 0.75" in text


def test_cli_renderer_response_no_text() -> None:
    renderer, buffer = _renderer()

    renderer.show_response(_orchestrator(agent_uid=None), None)

    text = _output(buffer)
    assert "Agent" in text
    assert "[No text output]" in text


def test_cli_renderer_history() -> None:
    renderer, buffer = _renderer()

    renderer.show_history(
        "# Transcript history\n\n## Turn 1\n\nThe posterior probability is 0.82.\n"
    )

    text = _output(buffer)
    assert "Transcript history" in text
    assert "Turn 1" in text
    assert "posterior probability is 0.82" in text


def test_cli_renderer_history_empty() -> None:
    renderer, buffer = _renderer()

    renderer.show_history("")

    assert "Transcript history is empty for this selection." in _output(buffer)


def test_cli_renderer_context() -> None:
    renderer, buffer = _renderer()

    renderer.show_context(
        "Memory context",
        "# Prior notes\n\nBaseline prior odds are 2:1.\n",
    )
    renderer.show_text_context(
        "Scratchpad context",
        "Compare prior scales of 1 and 2.",
    )
    renderer.show_text_context("Empty context", "")

    text = _output(buffer)
    assert "Memory context" in text
    assert "Prior notes" in text
    assert "Baseline prior odds are 2:1" in text
    assert "Scratchpad context" in text
    assert "Compare prior scales of 1 and 2." in text
    assert "[empty]" in text


def test_cli_renderer_team_context() -> None:
    renderer, buffer = _renderer()

    renderer.show_team_context(_team().to_context())

    text = _output(buffer)
    assert "Team context" in text
    assert "bayesian-analyst" in text
    assert "Bayesian analyst" in text
    assert "decision-analyst" in text
    assert "Decision analyst" in text


def test_cli_renderer_team_context_empty() -> None:
    renderer, buffer = _renderer()

    renderer.show_team_context({})

    assert "Team context is empty." in _output(buffer)


def test_cli_renderer_identity_context() -> None:
    renderer, buffer = _renderer()

    renderer.show_identity_context(
        "bayesian-analyst",
        {
            "name": "Bayesian analyst",
            "description": "Evaluates uncertainty using Bayesian inference.",
        },
    )

    text = _output(buffer)
    assert "Identity context" in text
    assert "bayesian-analyst" in text
    assert "Bayesian analyst" in text
    assert "Evaluates uncertainty using Bayesian inference." in text


def test_cli_renderer_context_files() -> None:
    renderer, buffer = _renderer()

    renderer.show_context_files(
        "Memory context",
        {
            "memory/prior_notes.md": "Baseline prior odds are 2:1.",
            "memory/robustness.txt": "The conclusion is stable.",
        },
    )

    text = _output(buffer)
    assert "Memory context" in text
    assert "memory/prior_notes.md" in text
    assert "Baseline prior odds are 2:1." in text
    assert "memory/robustness.txt" in text
    assert "The conclusion is stable." in text


def test_cli_renderer_context_files_empty() -> None:
    renderer, buffer = _renderer()

    renderer.show_context_files("Memory context", {})

    assert "Memory context is empty." in _output(buffer)


def test_cli_renderer_status() -> None:
    renderer, buffer = _renderer()
    state = InteractionState()
    state.transcript_history.append(
        TranscriptRound(
            agent_uid="bayesian-analyst",
            user_prompt="Estimate the posterior probability.",
            agent_output="The posterior probability is 0.82.",
        )
    )
    state.provider_history = _provider_history()

    renderer.show_status(_orchestrator(state=state))

    text = _output(buffer)
    assert "Status" in text
    assert "bayesian-analyst" in text
    assert "gpt-analysis" in text
    assert "gpt-5.6-luna" in text
    assert "Transcript rounds" in text
    assert "1" in text
    assert "Provider rounds" in text
    assert "2" in text
    assert "Current round" in text
    assert "none" in text


def test_cli_renderer_state() -> None:
    renderer, buffer = _renderer()
    state = InteractionState()
    state.transcript_history.append(
        TranscriptRound(
            agent_uid="decision-analyst",
            uid="7c9e6679-7425-40de-944b-e07fc1f90ae7",
            user_prompt="Compare the expected utility of the available actions.",
            agent_output="Action A has the higher expected utility.",
        )
    )
    state.provider_history = _provider_history()

    renderer.show_state(state)

    text = _output(buffer)
    assert "Interaction state" in text
    assert "Last transcript UID" in text
    assert "7c9e6679-7425-40de-944b-e07fc1f90ae7" in text
    assert "Last agent" in text
    assert "decision-analyst" in text
    assert "Provider rounds" in text
    assert "2" in text


def test_cli_renderer_provider_status() -> None:
    renderer, buffer = _renderer()

    renderer.show_provider_status(_provider_history())

    text = _output(buffer)
    assert "Provider history" in text
    assert "bayesian-analyst" in text
    assert "gpt-analysis" in text
    assert "gpt / gpt-5.6-luna" in text
    assert "decision-analyst" in text
    assert "gemini / gemini-3.5-flash-lite" in text
    assert "resp_bayesian_001" not in text
    assert "interaction_decision_001" not in text


def test_cli_renderer_provider_status_empty() -> None:
    renderer, buffer = _renderer()

    renderer.show_provider_status(ProviderHistory())

    assert "Provider history is empty." in _output(buffer)


def test_cli_renderer_info_error() -> None:
    renderer, buffer = _renderer()

    renderer.show_info("Bayesian analysis completed.")
    renderer.show_error("The selected agent has no engine mapping.")

    text = _output(buffer)
    assert "Bayesian analysis completed." in text
    assert "Error" in text
    assert "The selected agent has no engine mapping." in text


def test_cli_renderer_round_status() -> None:
    state = InteractionState()

    assert CLIRenderer._round_status(state) == "none"

    state.start_round(
        agent_uid="bayesian-analyst",
        team_context=_team().to_context(),
        user_prompt="Estimate the posterior probability under the baseline prior.",
    )
    assert CLIRenderer._round_status(state) == "incomplete"

    assert state.round is not None
    state.round.transcript.complete()
    assert CLIRenderer._round_status(state) == "completed"


def test_cli_renderer_redact_secrets() -> None:
    config = {
        "api_key": "secret-key",
        "reasoning_effort": "high",
        "nested": {
            "access_token": "secret-token",
            "options": [
                {"password": "secret-password"},
                {"thinking_level": "high"},
            ],
        },
    }

    redacted = CLIRenderer._redact_secrets(config)

    assert redacted == {
        "api_key": "[redacted]",
        "reasoning_effort": "high",
        "nested": {
            "access_token": "[redacted]",
            "options": [
                {"password": "[redacted]"},
                {"thinking_level": "high"},
            ],
        },
    }
    assert config["api_key"] == "secret-key"
    assert config["nested"]["access_token"] == "secret-token"
