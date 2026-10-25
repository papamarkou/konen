import argparse

from .session import CLISession


def main() -> None:
    """Launch the Konen interactive command-line interface."""
    parser = argparse.ArgumentParser(
        prog="konen", description="Human-centered orchestration of AI agents."
    )
    parser.add_argument(
        "workspace",
        nargs="?",
        default=".",
        help="Workspace root path (default: current directory).",
    )
    args = parser.parse_args()

    cli = CLISession.from_path(args.workspace)
    cli.run()


if __name__ == "__main__":
    main()
