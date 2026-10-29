"""Checkpoint contract shared by random initialization and future trained models."""

from pathlib import Path

import torch

from .model import AstLSTM

PAD = "<PAD>"
UNK = "<UNK>"
REPRESENTATION = "joern-ast-preorder-vars-v2"
CLASSES = ["safe", "vulnerable"]
DEMO_TOKENS = [  # TODO: replace with real tokens from training set
    "END",
    "METHOD",
    "METHOD_PARAMETER_IN",
    "METHOD_RETURN",
    "BLOCK",
    "IDENTIFIER",
    "LOCAL",
    "METHOD_PARAMETER_IN:VAR_1",
    "METHOD_PARAMETER_IN:VAR_2",
    "LOCAL:VAR_3",
    "IDENTIFIER:VAR_1",
    "IDENTIFIER:VAR_2",
    "IDENTIFIER:VAR_3",
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


def create_demo_checkpoint(path: Path, seed: int = 42) -> None:  # TODO: delete this function after training model
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
    """Read the checkpoint, validate its contents and build the model."""
    checkpoint = _read_checkpoint(path)
    _validate_metadata(checkpoint)
    _validate_vocabulary(checkpoint.get("vocab"))
    _validate_max_length(checkpoint.get("max_length"))
    _validate_architecture(checkpoint.get("architecture"))
    model = _build_model_from_checkpoint(checkpoint)
    return model, checkpoint


def _read_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a checkpoint dictionary.")
    return checkpoint


def _validate_metadata(checkpoint: dict) -> None:
    if checkpoint.get("format_version") != 1 or checkpoint.get("representation") != REPRESENTATION:
        raise ValueError("Unsupported checkpoint format or AST representation.")
    if checkpoint.get("classes") != CLASSES or type(checkpoint.get("trained")) is not bool:
        raise ValueError("Checkpoint must declare classes and a boolean trained flag.")


def _validate_vocabulary(vocab: dict[str, int]) -> None:
    message = "Checkpoint vocabulary must use contiguous IDs with PAD=0 and UNK=1."
    if not isinstance(vocab, dict) or vocab.get(PAD) != 0 or vocab.get(UNK) != 1:
        raise ValueError(message)
    if any(not isinstance(token, str) or type(index) is not int for token, index in vocab.items()):
        raise ValueError(message)
    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError(message)


def _validate_max_length(max_length: int) -> None:
    if type(max_length) is not int or max_length < 1:
        raise ValueError("Checkpoint max_length must be a positive integer.")


def _validate_architecture(architecture: dict[str, int]) -> None:
    message = "Checkpoint must specify positive embedding_dim and hidden_dim."
    if not isinstance(architecture, dict) or set(architecture) != {"embedding_dim", "hidden_dim"}:
        raise ValueError(message)
    if any(type(value) is not int or value < 1 for value in architecture.values()):
        raise ValueError(message)


def _build_model_from_checkpoint(checkpoint: dict) -> AstLSTM:
    model = AstLSTM(len(checkpoint["vocab"]), **checkpoint["architecture"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval()
