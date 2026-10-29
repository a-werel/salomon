"""Container CLI: source code -> Joern AST -> checkpoint -> JSON."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from .checkpoint import CLASSES, REPRESENTATION, UNK, create_demo_checkpoint, load_checkpoint

DEFAULT_CHECKPOINT = Path("/app/checkpoints/model.pt")


def export_sequences(input_path: Path, timeout: int = 600) -> list[dict]:
    if not input_path.exists():
        raise ValueError(f"Source path does not exist: {input_path}")
    script = Path(__file__).with_name("export_sequences.sc")
    with tempfile.TemporaryDirectory(prefix="salomon-ast-") as directory:
        work = Path(directory)
        output = work / "sequences.json"
        subprocess.run(
            [
                "joern",
                "--script",
                str(script),
                "--param",
                f"inputPath={input_path.resolve()}",
                "--param",
                f"outputPath={output}",
            ],
            cwd=work,
            stdout=sys.stderr,
            stderr=sys.stderr,
            check=True,
            timeout=timeout,
        )
        return json.loads(output.read_text(encoding="utf-8"))


def predict_records(records: list[dict], checkpoint_path: Path) -> dict:
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a nonempty list of function AST sequences.")
    model, checkpoint = load_checkpoint(checkpoint_path)
    vocab, limit, trained = checkpoint["vocab"], checkpoint["max_length"], checkpoint["trained"]
    predictions = []
    with torch.inference_mode():
        for record in records:
            if not isinstance(record, dict):
                raise TypeError("Each AST record must be an object.")
            tokens = record.get("tokens")
            if not isinstance(tokens, list) or not tokens or any(not isinstance(token, str) for token in tokens):
                raise ValueError("Each function must have a nonempty list of string tokens.")
            if not isinstance(record.get("source"), str) or not isinstance(record.get("function"), str):
                raise TypeError("Each function must have source and function names.")
            used = tokens[:limit]
            ids = [vocab.get(token, vocab[UNK]) for token in used]
            logits = model(torch.tensor([ids]), torch.tensor([len(ids)]))
            scores = logits.softmax(dim=1)[0].tolist()
            predictions.append(
                {
                    "source": record["source"],
                    "function": record["function"],
                    "prediction": CLASSES[max(range(len(scores)), key=scores.__getitem__)] if trained else None,
                    "scores": dict(zip(CLASSES, scores, strict=True)),
                    "token_count": len(tokens),
                    "used_token_count": len(used),
                    "unknown_token_count": sum(token not in vocab for token in used),
                    "truncated": len(tokens) > limit,
                }
            )
    result = {
        "model_status": "trained" if trained else "untrained",
        "representation": REPRESENTATION,
        "assessment_available": trained,
        "predictions": predictions,
    }
    if not trained:
        result["warning"] = "Random weights: scores are for integration testing, not vulnerability assessment."
    return result


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init-demo", help="Create explicitly untrained random weights.")
    initialize.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT)
    initialize.add_argument("--seed", type=int, default=42)
    for name in ("predict", "predict-sequences"):
        command = commands.add_parser(name)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
        if name == "predict":
            command.add_argument("--timeout", type=positive_int, default=600)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        if args.command == "init-demo":
            create_demo_checkpoint(args.output, args.seed)
            print(json.dumps({"checkpoint": str(args.output), "model_status": "untrained"}))
            return
        records = (
            export_sequences(args.input, args.timeout)
            if args.command == "predict"
            else json.loads(args.input.read_text(encoding="utf-8"))
        )
        print(json.dumps(predict_records(records, args.checkpoint)))
    except (OSError, ValueError, TypeError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(1, f"LSTM inference failed: {error}\n")


if __name__ == "__main__":
    main()
