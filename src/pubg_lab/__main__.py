"""Discover and explicitly run a registered research application."""

import argparse
import json
from pathlib import Path
import runpy
import sys


def command_catalog() -> dict:
    return json.loads(Path(__file__).with_name("commands.json").read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pubg-lab", description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("list", help="List commands without importing or executing them")
    run = subparsers.add_parser("run", help="Run one explicitly selected application")
    run.add_argument("command")
    run.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    commands = command_catalog()
    if args.action == "list":
        for key, record in sorted(commands.items()):
            print(f"{key:62} {record['module']}")
        return 0
    if args.command not in commands:
        parser.error(f"unknown command: {args.command}; use 'pubg-lab list'")
    record = commands[args.command]
    sys.argv = [record["module"], *args.arguments]
    runpy.run_module(record["module"], run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
