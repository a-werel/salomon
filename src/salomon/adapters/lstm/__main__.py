"""Run Docker inference from the host without installing PyTorch."""

import argparse
import json
from pathlib import Path

from .adapter import LSTMAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="C/C++ source file or directory.")
    parser.add_argument("--image", default="salomon-lstm:latest")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    try:
        result = LSTMAdapter(image=args.image, checkpoint=args.checkpoint).analyze_code(args.input)
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
