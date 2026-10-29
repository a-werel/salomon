"""Checkpoint contract shared by random initialization and future trained models."""

from pathlib import Path

import torch

from .model import AstLSTM

PAD = "<PAD>"
UNK = "<UNK>"
REPRESENTATION = "joern-ast-preorder-v1"
CLASSES = ["safe", "vulnerable"]
DEMO_TOKENS = [
    "END",
    "METHOD",
    "METHOD_PARAMETER_IN",
    "METHOD_RETURN",
    "BLOCK",
    "IDENTIFIER",
    "LOCAL",
    "RETURN",
    "FIELD_IDENTIFIER",
    "TYPE_REF",
    "JUMP_TARGET",
    "CONTROL:IF",
    "CONTROL:ELSE",
    "CONTROL:FOR",
    "CONTROL:WHILE",
    "CONTROL:SWITCH",
    "CALL:<operator>.assignment",
    "CALL:<operator>.addition",
    "CALL:<operator>.subtraction",
    "CALL:<operator>.multiplication",
    "CALL:<operator>.division",
    "CALL:<operator>.equals",
    "CALL:<operator>.notEquals",
    "CALL:<operator>.lessThan",
    "CALL:<operator>.greaterThan",
    "CALL:<operator>.lessEqualsThan",
    "CALL:<operator>.greaterEqualsThan",
    "CALL:<operator>.logicalAnd",
    "CALL:<operator>.logicalOr",
    "CALL:<operator>.logicalNot",
    "CALL:<operator>.addressOf",
    "CALL:<operator>.indirection",
    "CALL:<operator>.indirectIndexAccess",
    "CALL:<operator>.cast",
    "CALL:<operator>.sizeOf",
    "CALL:strcpy",
    "CALL:strncpy",
    "CALL:strlen",
    "CALL:memcpy",
    "CALL:malloc",
    "CALL:free",
    "CALL:printf",
    "LITERAL:0",
    "LITERAL:1",
    "LITERAL:NULL",
]


def create_demo_checkpoint(path: Path, seed: int = 42) -> None:
    """Create reproducible random weights, never labelled as a trained detector."""
    vocab = {token: index for index, token in enumerate([PAD, UNK, *sorted(set(DEMO_TOKENS))])}
    architecture = {"embedding_dim": 32, "hidden_dim": 64}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = AstLSTM(len(vocab), **architecture)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "representation": REPRESENTATION,
            "trained": False,
            "seed": seed,
            "classes": CLASSES,
            "architecture": architecture,
            "vocab": vocab,
            "max_length": 512,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path: Path) -> tuple[AstLSTM, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a checkpoint dictionary.")
    if checkpoint.get("format_version") != 1 or checkpoint.get("representation") != REPRESENTATION:
        raise ValueError("Unsupported checkpoint format or AST representation.")
    if checkpoint.get("classes") != CLASSES or type(checkpoint.get("trained")) is not bool:
        raise ValueError("Checkpoint must declare classes and a boolean trained flag.")
    vocab = checkpoint.get("vocab")
    if (
        not isinstance(vocab, dict)
        or any(not isinstance(token, str) or type(index) is not int for token, index in vocab.items())
        or vocab.get(PAD) != 0
        or vocab.get(UNK) != 1
        or sorted(vocab.values()) != list(range(len(vocab)))
    ):
        raise ValueError("Checkpoint vocabulary must use contiguous IDs with PAD=0 and UNK=1.")
    if type(checkpoint.get("max_length")) is not int or checkpoint["max_length"] < 1:
        raise ValueError("Checkpoint max_length must be a positive integer.")
    architecture = checkpoint.get("architecture")
    if (
        not isinstance(architecture, dict)
        or set(architecture) != {"embedding_dim", "hidden_dim"}
        or any(type(value) is not int or value < 1 for value in architecture.values())
    ):
        raise ValueError("Checkpoint must specify positive embedding_dim and hidden_dim.")
    model = AstLSTM(len(vocab), **architecture)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval(), checkpoint
