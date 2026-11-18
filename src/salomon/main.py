import argparse
import os
import subprocess
import sys
from pathlib import Path

from salomon.adapters.vulcnn.adapter import VulCNNAdapter


def main() -> int:
    parser = argparse.ArgumentParser(description="Create PDGs or classify C/C++ source with VulCNN.")
    parser.add_argument("--workspace", default=os.environ.get("VULCNN_WORKSPACE", "outputs/vulcnn"))
    parser.add_argument("--models", default=os.environ.get("VULCNN_MODELS", "models"))
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("pdg", "analyze"):
        command = commands.add_parser(name)
        command.add_argument("source", type=Path)
    args = parser.parse_args()
    adapter = VulCNNAdapter(workspace=args.workspace, models=args.models)
    try:
        source = args.source.read_bytes()
        if args.command == "pdg":
            for path in adapter.create_pdg(source, filename=args.source.name):
                print(path)
        else:
            print(adapter.analyze_code(source, filename=args.source.name))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr, file=sys.stderr, end="")
        print(f"Analysis failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
