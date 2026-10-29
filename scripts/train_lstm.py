"""Train the canonical AstLSTM separately from the inference container."""

import argparse
import json
import math
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from salomon.adapters.lstm.adapter import bind_mount
from salomon.adapters.lstm.checkpoint import CLASSES, PAD, REPRESENTATION, UNK, load_checkpoint
from salomon.adapters.lstm.model import AstLSTM


@dataclass
class Example:
    source: str
    function: str
    tokens: list[str]
    label: int
    split: str
    group: str


def read_json(path: Path) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records or any(not isinstance(record, dict) for record in records):
        raise ValueError(f"Expected a nonempty JSON array of objects: {path}")
    return records


def export_with_docker(source: Path, image: str, timeout: int) -> list[dict]:
    source = source.resolve(strict=True)
    program = (
        "import json, sys; from pathlib import Path; "
        "from salomon.adapters.lstm.worker import export_sequences; "
        "print(json.dumps(export_sequences(Path(sys.argv[1]), int(sys.argv[2]))))"
    )
    command = [
        "docker",
        "run",
        "--rm",
        "--mount",
        bind_mount(source.parent, "/input"),
        "--entrypoint",
        "python3",
        image,
        "-c",
        program,
        f"/input/{source.name}",
        str(timeout),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=timeout + 30)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"AST export failed: {error.stderr.strip()}") from error
    return json.loads(result.stdout)


def labeled_examples(records: list[dict], labels: list[dict]) -> list[Example]:
    if not isinstance(records, list) or not records:
        raise ValueError("No function AST sequences were exported.")
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("An AST record must be an object.")
        if record.get("representation") != REPRESENTATION:
            raise ValueError("AST representation is missing or outdated; export sequences again with the current image.")
        if not isinstance(record.get("source"), str) or not isinstance(record.get("function"), str):
            raise TypeError("AST records must contain source and function names.")
        tokens = record.get("tokens")
        if not isinstance(tokens, list) or not tokens or any(not isinstance(token, str) for token in tokens):
            raise ValueError("AST records must contain nonempty lists of string tokens.")

    examples = []
    used_records = set()
    group_splits = {}
    sequence_splits = {}
    for entry in labels:
        source, function = entry.get("source"), entry.get("function")
        if not isinstance(source, str) or not source or not isinstance(function, str) or not function:
            raise ValueError("Labels must contain nonempty source and function names.")
        source = source.replace("\\", "/")
        label, split = entry.get("label"), entry.get("split")
        if type(label) is not int or label not in (0, 1) or split not in ("train", "validation", "test"):
            raise ValueError("Each label needs label=0/1 and split=train/validation/test.")
        group = entry.get("group", source)
        if not isinstance(group, str) or not group:
            raise ValueError("A group must be a nonempty string.")
        if group in group_splits and group_splits[group] != split:
            raise ValueError(f"Related examples in group {group!r} span multiple splits.")
        group_splits[group] = split
        matches = [
            index
            for index, record in enumerate(records)
            if record["function"] == function
            and (
                record["source"].replace("\\", "/") == source
                or record["source"].replace("\\", "/").endswith("/" + source)
            )
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one AST for {source}:{function}, found {len(matches)}.")
        index = matches[0]
        if index in used_records:
            raise ValueError(f"Duplicate label for {source}:{function}.")
        used_records.add(index)
        tokens = records[index]["tokens"]
        sequence = tuple(tokens)
        if sequence in sequence_splits and sequence_splits[sequence] != split:
            raise ValueError("Identical AST sequences occur in different splits.")
        sequence_splits[sequence] = split
        examples.append(Example(source, function, tokens, label, split, group))
    for split in ("train", "validation", "test"):
        selected = [example for example in examples if example.split == split]
        if split == "test" and not selected:
            continue
        if {example.label for example in selected} != {0, 1}:
            raise ValueError(f"The {split} split must contain both safe and vulnerable examples.")
    return examples


def build_vocabulary(examples: list[Example]) -> dict[str, int]:
    tokens = {token for example in examples if example.split == "train" for token in example.tokens}
    return {token: index for index, token in enumerate([PAD, UNK, *sorted(tokens - {PAD, UNK})])}


def make_batch(
    examples: list[Example], vocab: dict[str, int], max_length: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences = [
        torch.tensor([vocab.get(token, vocab[UNK]) for token in item.tokens[:max_length]]) for item in examples
    ]
    tokens = pad_sequence(sequences, batch_first=True, padding_value=vocab[PAD])
    lengths = torch.tensor([len(sequence) for sequence in sequences])
    labels = torch.tensor([example.label for example in examples])
    return tokens, lengths, labels


def train_epoch(
    model: AstLSTM,
    examples: list[Example],
    vocab: dict[str, int],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    max_length: int,
    rng: random.Random,
) -> float:
    model.train()
    shuffled = rng.sample(examples, len(examples))
    total_loss = 0.0
    for start in range(0, len(shuffled), batch_size):
        tokens, lengths, labels = make_batch(shuffled[start : start + batch_size], vocab, max_length)
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(tokens, lengths), labels)
        if not torch.isfinite(loss):
            raise ValueError("Training produced a non-finite loss; no checkpoint was saved for this epoch.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(examples)


def evaluate(
    model: AstLSTM,
    examples: list[Example],
    vocab: dict[str, int],
    batch_size: int,
    max_length: int,
) -> dict:
    model.eval()
    total_loss = 0.0
    tp = fp = fn = tn = 0
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            tokens, lengths, labels = make_batch(examples[start : start + batch_size], vocab, max_length)
            logits = model(tokens, lengths)
            total_loss += torch.nn.functional.cross_entropy(logits, labels).item() * len(labels)
            predictions = logits.argmax(dim=1)
            tp += int(((predictions == 1) & (labels == 1)).sum())
            fp += int(((predictions == 1) & (labels == 0)).sum())
            fn += int(((predictions == 0) & (labels == 1)).sum())
            tn += int(((predictions == 0) & (labels == 0)).sum())
    return {
        "samples": len(examples),
        "loss": total_loss / len(examples),
        "accuracy": (tp + tn) / len(examples),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def save_checkpoint(path: Path, model: AstLSTM, vocab: dict[str, int], args: argparse.Namespace, epoch: int) -> None:
    torch.save(
        {
            "format_version": 1,
            "representation": REPRESENTATION,
            "trained": True,
            "classes": CLASSES,
            "architecture": {"embedding_dim": args.embedding_dim, "hidden_dim": args.hidden_dim},
            "vocab": vocab,
            "max_length": args.max_length,
            "state_dict": model.state_dict(),
            "seed": args.seed,
            "epoch": epoch,
        },
        path,
    )


def train(examples: list[Example], args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise ValueError(f"Output already exists; choose another path to preserve its weights: {args.output}")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = random.Random(args.seed)
    splits = {
        name: [example for example in examples if example.split == name] for name in ("train", "validation", "test")
    }
    vocab = build_vocabulary(examples)
    model = AstLSTM(len(vocab), embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, splits["train"], vocab, optimizer, args.batch_size, args.max_length, rng)
        validation = evaluate(model, splits["validation"], vocab, args.batch_size, args.max_length)
        if not math.isfinite(validation["loss"]):
            raise ValueError("Validation produced a non-finite loss.")
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            save_checkpoint(args.output, model, vocab, args, epoch)
        history.append({"epoch": epoch, "train_loss": loss, "validation": validation})
        print(f"epoch={epoch:03d} train_loss={loss:.4f} validation_loss={validation['loss']:.4f}", flush=True)
    model, checkpoint = load_checkpoint(args.output)
    metrics = {
        "selected_epoch": checkpoint["epoch"],
        "vocabulary_size": len(vocab),
        "split_sizes": {name: len(items) for name, items in splits.items()},
        "truncated_functions": sum(len(example.tokens) > args.max_length for example in examples),
        "train": evaluate(model, splits["train"], vocab, args.batch_size, args.max_length),
        "validation": evaluate(model, splits["validation"], vocab, args.batch_size, args.max_length),
        "test": evaluate(model, splits["test"], vocab, args.batch_size, args.max_length) if splits["test"] else None,
        "history": history,
    }
    args.output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Must be a finite positive number.")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="C/C++ source file or directory; exports AST using Docker.")
    source.add_argument("--sequences", type=Path, help="Previously exported function AST JSON; no Docker needed.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/lstm/trained.pt"))
    parser.add_argument("--image", default="salomon-lstm:latest")
    parser.add_argument("--epochs", type=positive_int, default=20)
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--max-length", type=positive_int, default=512)
    parser.add_argument("--embedding-dim", type=positive_int, default=32)
    parser.add_argument("--hidden-dim", type=positive_int, default=64)
    parser.add_argument("--learning-rate", type=positive_float, default=0.003)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-timeout", type=positive_int, default=600)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError(f"Output already exists: {args.output}; choose another --output path.")
        labels = read_json(args.labels)
        records = (
            read_json(args.sequences)
            if args.sequences
            else export_with_docker(args.input, args.image, args.export_timeout)
        )
        examples = labeled_examples(records, labels)
        print(
            "Labels are supplied by you. Small demo data does not establish vulnerability detection accuracy.",
            flush=True,
        )
        metrics = train(examples, args)
        if not args.sequences:
            args.output.with_suffix(".sequences.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {"checkpoint": str(args.output), "selected_epoch": metrics["selected_epoch"], "test": metrics["test"]},
                indent=2,
            )
        )
    except (OSError, ValueError, TypeError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(1, f"LSTM training failed: {error}\n")


if __name__ == "__main__":
    main()
