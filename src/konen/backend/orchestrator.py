from .ai_team import AITeam
from .engine_router import EngineRouter
from .interaction_state import InteractionState
from .llm_pool import LLMPool
from .provider_round import ProviderRound
from .transcript_round import TranscriptRound
from .workspace import Workspace


class Orchestrator:
    """Coordinate human-AI interaction."""

    def __init__(
        self,
        workspace: Workspace | None = None,
        ai_team: AITeam | None = None,
        llm_pool: LLMPool | None = None,
        engine_router: EngineRouter | None = None,
        agent_uid: str | None = None,
        interaction_state: InteractionState | None = None,
    ) -> None:
        self.workspace = Workspace() if workspace is None else workspace
        self.ai_team = AITeam() if ai_team is None else ai_team
        self.llm_pool = LLMPool() if llm_pool is None else llm_pool
        self.engine_router = EngineRouter() if engine_router is None else engine_router
        self.agent_uid = agent_uid
        self.interaction_state = (
            InteractionState() if interaction_state is None else interaction_state
        )

    @classmethod
    def from_workspace(
        cls,
        workspace: Workspace,
        agent_uid: str | None = None,
        interaction_state: InteractionState | None = None,
    ) -> "Orchestrator":
        """Build an orchestrator from workspace configuration."""
        workspace.ensure_initialized()

        ai_team = AITeam.from_workspace(workspace)
        engine_router = EngineRouter.from_workspace(workspace)
        llm_pool = LLMPool.from_workspace(workspace)

        if agent_uid is not None:
            ai_team.get_agent(agent_uid)

        return cls(
            workspace=workspace,
            ai_team=ai_team,
            llm_pool=llm_pool,
            engine_router=engine_router,
            agent_uid=agent_uid,
            interaction_state=interaction_state,
        )

    @property
    def engine_uid(self) -> str:
        """Return the engine identifier assigned to the selected agent."""
        if self.agent_uid is None:
            raise ValueError(
                "Orchestrator.agent_uid must be set to resolve engine UID."
            )
        return self.engine_router.get_engine_uid(self.agent_uid)

    def reload_from_workspace(self) -> None:
        """Reload orchestration state from workspace configuration."""
        ai_team = AITeam.from_workspace(self.workspace)
        engine_router = EngineRouter.from_workspace(self.workspace)
        llm_pool = LLMPool.from_workspace(self.workspace)

        agent_uid = self.agent_uid
        if agent_uid is not None and agent_uid not in ai_team.list_agents():
            agent_uid = None

        self.ai_team = ai_team
        self.engine_router = engine_router
        self.llm_pool = llm_pool
        self.agent_uid = agent_uid

    def run_turn(
        self,
        user_prompt: str,
        history_limit: int | None = None,
        memory_filenames: tuple[str, ...] | None = None,
        scratchpad_agent_uids: tuple[str, ...] | None = None,
    ) -> str | None:
        """Run one user-agent interaction turn."""
        self.workspace.ensure_initialized()

        if self.agent_uid is None:
            raise ValueError(
                "Orchestrator.agent_uid must be set before running a turn."
            )

        if history_limit is not None and history_limit < 0:
            raise ValueError("history_limit must be non-negative or None.")

        agent_uid = self.agent_uid

        self.ai_team.get_agent(agent_uid)

        engine_uid = self.engine_router.get_engine_uid(agent_uid)
        engine = self.llm_pool.get_engine(engine_uid)

        selected_history = self.interaction_state.transcript_history.tail(history_limit)

        selected_history_start_uid = (
            None
            if selected_history.first_round is None
            else selected_history.first_round.uid
        )

        previous_provider_round = self.interaction_state.provider_history.last_for(
            agent_uid=agent_uid,
            engine_uid=engine_uid,
            model_family=engine.model_family,
            model_name=engine.model_name,
        )

        can_continue = (
            previous_provider_round is not None
            and selected_history_start_uid is not None
            and (
                previous_provider_round.history_start_uid == selected_history_start_uid
            )
            and self.interaction_state.transcript_history.contains(
                previous_provider_round.transcript_uid
            )
        )

        if can_continue:
            prompt_history = self.interaction_state.transcript_history.after(
                previous_provider_round.transcript_uid
            )

            provider_chain = self.interaction_state.provider_history.for_chain(
                agent_uid=agent_uid,
                engine_uid=engine_uid,
                model_family=engine.model_family,
                model_name=engine.model_name,
                history_start_uid=(previous_provider_round.history_start_uid),
            )

            if (
                not provider_chain
                or provider_chain[-1].transcript_uid
                != previous_provider_round.transcript_uid
            ):
                raise RuntimeError(
                    "Provider continuation chain is inconsistent with its "
                    "latest provider round."
                )

            provider_state = engine.prepare_provider_state(
                [round.state for round in provider_chain]
            )
            provider_history_start_uid = previous_provider_round.history_start_uid
        else:
            prompt_history = selected_history
            provider_state = None
            provider_history_start_uid = None

        team_context = self.ai_team.to_context()
        memory_context = self.workspace.load_memory_context(memory_filenames)
        selected_scratchpads = (
            (agent_uid,) if scratchpad_agent_uids is None else scratchpad_agent_uids
        )
        scratchpad_context: dict[str, str] = {}
        for scratchpad_agent_uid in selected_scratchpads:
            scratchpad_context.update(
                self.workspace.load_scratchpad_context(scratchpad_agent_uid)
            )

        self.interaction_state.start_round(
            agent_uid=agent_uid,
            team_context=team_context,
            user_prompt=user_prompt,
            memory_context=memory_context,
            scratchpad_context=scratchpad_context,
            metadata={"history_limit": history_limit},
        )

        interaction_round = self.interaction_state.round
        if interaction_round is None:
            raise RuntimeError("Interaction round was not initialized.")

        system_instruction = interaction_round.build_prompt_context()

        prompt = interaction_round.build_prompt_input(
            history=prompt_history, history_limit=None
        )

        response = engine.generate_response(
            prompt=prompt,
            system_instruction=system_instruction,
            provider_state=provider_state,
        )

        transcript = interaction_round.transcript
        transcript.agent_output = response.text
        transcript.update_metadata(response.metadata)
        transcript.complete()

        self.interaction_state.append_to_history()

        if response.provider_state:
            if provider_history_start_uid is None:
                if prompt_history.first_round is not None:
                    provider_history_start_uid = prompt_history.first_round.uid
                else:
                    provider_history_start_uid = transcript.uid

            provider_round = ProviderRound(
                transcript_uid=transcript.uid,
                agent_uid=agent_uid,
                engine_uid=engine_uid,
                model_family=engine.model_family,
                model_name=engine.model_name,
                history_start_uid=provider_history_start_uid,
                state=response.provider_state,
            )

            self.interaction_state.append_provider_round(provider_round)

        return response.text

    def append_last_turn_to_memory(self, filename: str) -> None:
        """Append the latest transcript turn to a memory file."""
        transcript = self._last_transcript_round()
        self.workspace.append_to_memory(filename, transcript.to_markdown())

    def append_last_turn_to_scratchpad(self, agent_uid: str) -> None:
        """Append the latest transcript turn to an agent scratchpad."""
        transcript = self._last_transcript_round()
        self.ai_team.get_agent(agent_uid)
        self.workspace.append_to_scratchpad(agent_uid, transcript.to_markdown())

    def _last_transcript_round(self) -> TranscriptRound:
        """Return the latest transcript round."""
        transcript = self.interaction_state.transcript_history.last_round
        if transcript is None or transcript.completed_at is None:
            raise ValueError("No completed transcript turn is available.")
        return transcript
