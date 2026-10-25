from dataclasses import FrozenInstanceError
from uuid import UUID

import pytest

from konen import AIAgent


def test_ai_agent_preserves_fields() -> None:
    agent = AIAgent(
        name="Bayesian analyst",
        description="Evaluates uncertainty using Bayesian inference.",
        uid="bayesian-analyst",
    )

    assert agent.name == "Bayesian analyst"
    assert agent.description == "Evaluates uncertainty using Bayesian inference."
    assert agent.uid == "bayesian-analyst"


def test_ai_agent_default_uids() -> None:
    first = AIAgent(name="First", description="First agent.")
    second = AIAgent(name="Second", description="Second agent.")

    assert first.uid != second.uid
    assert UUID(first.uid).version == 4
    assert UUID(second.uid).version == 4


def test_ai_agent_is_immutable() -> None:
    agent = AIAgent(
        name="Bayesian analyst",
        description="Evaluates uncertainty using Bayesian inference.",
        uid="bayesian-analyst",
    )

    with pytest.raises(FrozenInstanceError):
        setattr(agent, "name", "Frequentist analyst")
