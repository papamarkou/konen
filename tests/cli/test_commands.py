from dataclasses import FrozenInstanceError

import pytest

from konen.cli.commands import (
    COMMAND_GROUPS,
    COMMAND_SPECS,
    CommandGroup,
    CommandSpec,
    commands_for_group,
    resolve_group,
    static_completions,
)


def test_command_group_preserves_fields() -> None:
    group = CommandGroup(
        key="analysis",
        title="Analysis",
        aliases=("bayesian", "decision"),
    )

    assert group.key == "analysis"
    assert group.title == "Analysis"
    assert group.aliases == ("bayesian", "decision")


def test_command_group_is_immutable() -> None:
    group = CommandGroup(key="analysis", title="Analysis")

    with pytest.raises(FrozenInstanceError):
        setattr(group, "title", "Updated analysis")


def test_command_spec_preserves_fields() -> None:
    spec = CommandSpec(
        group="agent",
        syntax="/agent select bayesian-analyst",
        description="Select the Bayesian analyst for subsequent prompts.",
        completion="/agent select",
    )

    assert spec.group == "agent"
    assert spec.syntax == "/agent select bayesian-analyst"
    assert spec.description == ("Select the Bayesian analyst for subsequent prompts.")
    assert spec.completion == "/agent select"


def test_command_spec_is_immutable() -> None:
    spec = CommandSpec(
        group="agent",
        syntax="/agent select bayesian-analyst",
        description="Select the Bayesian analyst for subsequent prompts.",
        completion="/agent select",
    )

    with pytest.raises(FrozenInstanceError):
        setattr(spec, "group", "session")


def test_command_groups_unique() -> None:
    keys = [group.key for group in COMMAND_GROUPS]
    names = [
        name.lower() for group in COMMAND_GROUPS for name in (group.key, *group.aliases)
    ]

    assert len(keys) == len(set(keys))
    assert len(names) == len(set(names))


def test_command_specs_valid_groups() -> None:
    group_keys = {group.key for group in COMMAND_GROUPS}

    assert COMMAND_SPECS
    assert all(spec.group in group_keys for spec in COMMAND_SPECS)


def test_command_specs_unique_completions() -> None:
    completions = [spec.completion for spec in COMMAND_SPECS]

    assert len(completions) == len(set(completions))


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("general", "general"),
        ("agent", "agent"),
        ("engine", "engine"),
        ("mapping", "mapping"),
        ("files", "files"),
        ("transcript", "transcript"),
        ("session", "session"),
    ],
)
def test_resolve_group_key(name: str, key: str) -> None:
    group = resolve_group(name)

    assert group is not None
    assert group.key == key


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("file", "files"),
        ("memory", "files"),
        ("scratchpad", "files"),
        ("history", "transcript"),
        ("turn", "transcript"),
        ("context", "transcript"),
        ("state", "session"),
        ("provider", "session"),
        ("history-limit", "session"),
    ],
)
def test_resolve_group_alias(name: str, key: str) -> None:
    group = resolve_group(name)

    assert group is not None
    assert group.key == key


def test_resolve_group_normalizes_input() -> None:
    group = resolve_group("  MeMoRy  ")

    assert group is not None
    assert group.key == "files"


def test_resolve_group_missing() -> None:
    assert resolve_group("posterior") is None


def test_commands_for_group() -> None:
    specs = commands_for_group("agent")

    assert specs
    assert all(spec.group == "agent" for spec in specs)
    assert [spec.syntax for spec in specs] == [
        "/agent list",
        "/agent show <uid>",
        "/agent select <uid>",
        "/agent add",
        "/agent remove <uid> [scratchpad]",
    ]


def test_commands_for_group_missing() -> None:
    assert commands_for_group("analysis") == ()


def test_context_selection_commands_in_files_help() -> None:
    syntaxes = [spec.syntax for spec in commands_for_group("files")]

    assert "/memory select [all|none|<filename> ...]" in syntaxes
    assert "/scratchpad select-next [@none|@active|<agent_uid> ...]" in syntaxes


def test_static_completions() -> None:
    completions = static_completions()

    assert completions == sorted(set(completions))
    assert "/exit" in completions
    assert "/agent select" in completions
    assert "/mapping assign" in completions
    assert "/memory append" in completions
    assert "/history agent" in completions
    assert "/provider clear" in completions

    for group in COMMAND_GROUPS:
        assert f"/help {group.key}" in completions

    for spec in COMMAND_SPECS:
        assert spec.completion in completions
