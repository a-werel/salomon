"""Check whether AstLSTM can memorize 100 complete, non-conflicting training examples."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from train_lstm import (
    Example,
    build_vocabulary,
    evaluate,
    labeled_examples,
    positive_int,
    read_json,
    save_checkpoint,
    train_epoch,
)

from salomon.adapters.lstm.checkpoint import load_checkpoint
from salomon.adapters.lstm.model import AstLSTM


def select_examples(examples: list[Example], seed: int, max_length: int) -> list[Example]:
    inputs = defaultdict(set)
    for example in examples:
        inputs[tuple(example.tokens)].add(example.label)
    eligible = [e for e in examples if len(e.tokens) <= max_length and len(inputs[tuple(e.tokens)]) == 1]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    selected, seen = [], set()
    for split, per_class in (("train", 50), ("validation", 10)):
        for label in (0, 1):
            candidates = []
            for example in eligible:
                identity = tuple(example.tokens)
                if example.split == split and example.label == label and identity not in seen:
                    candidates.append(example)
                    seen.add(identity)
                    if len(candidates) == per_class:
                        break
            if len(candidates) != per_class:
                raise ValueError(f"Not enough complete, distinct examples for {split}, label={label}.")
            selected.extend(candidates)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/lstm/learning-100-v2.pt"))
    parser.add_argument("--epochs", type=positive_int, default=100)
    parser.add_argument("--max-length", type=positive_int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Checkpoint already exists; choose another --output.")
    examples = labeled_examples(read_json(args.dataset / "sequences.json"), read_json(args.dataset / "labels.json"))
    selected = select_examples(examples, args.seed, args.max_length)
    training = [e for e in selected if e.split == "train"]
    validation = [e for e in selected if e.split == "validation"]
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    rng = random.Random(args.seed)
    vocab = build_vocabulary(selected)
    args.embedding_dim, args.hidden_dim = 32, 64
    model = AstLSTM(len(vocab), args.embedding_dim, args.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".labels.json").write_text(
        json.dumps([{k: v for k, v in vars(e).items() if k != "tokens"} for e in selected], indent=2), encoding="utf-8"
    )
    best_loss, history = float("inf"), []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, training, vocab, optimizer, 16, args.max_length, rng)
        metrics = evaluate(model, training, vocab, 16, args.max_length)
        history.append({"epoch": epoch, "train_loss": loss, "train": metrics})
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            save_checkpoint(args.output, model, vocab, args, epoch)
        print(f"epoch={epoch:03d} train_loss={loss:.4f} train_accuracy={metrics['accuracy']:.1%}", flush=True)
        if metrics["accuracy"] == 1.0:
            break
    model, checkpoint = load_checkpoint(args.output)
    report = {
        "purpose": "memorization check; checkpoint selected by training loss, not validation",
        "seed": args.seed,
        "selected_epoch": checkpoint["epoch"],
        "epochs_run": len(history),
        "max_length": args.max_length,
        "learning_rate": 0.01,
        "batch_size": 16,
        "train": evaluate(model, training, vocab, 16, args.max_length),
        "validation": evaluate(model, validation, vocab, 16, args.max_length),
        "history": history,
    }
    args.output.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
