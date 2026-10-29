import pytest

torch = pytest.importorskip("torch", reason="Install the lstm extra to test the model runtime.")

from salomon.adapters.lstm.checkpoint import UNK, create_demo_checkpoint, load_checkpoint
from salomon.adapters.lstm.worker import predict_records


@pytest.fixture
def checkpoint(tmp_path):
    torch.set_num_threads(1)
    path = tmp_path / "model.pt"
    create_demo_checkpoint(path)
    return path


def test_random_checkpoint_is_reproducible_and_never_returns_a_security_verdict(checkpoint, tmp_path):
    second = tmp_path / "second.pt"
    create_demo_checkpoint(second)
    records = [{"source": "example.c", "function": "example", "tokens": ["METHOD", "END"]}]
    result = predict_records(records, checkpoint)
    assert result == predict_records(records, second)
    assert result["model_status"] == "untrained"
    assert result["assessment_available"] is False
    assert result["predictions"][0]["prediction"] is None
    assert sum(result["predictions"][0]["scores"].values()) == pytest.approx(1)
    assert "Random weights" in result["warning"]


def test_external_checkpoint_controls_weights_vocabulary_and_sequence_limit(checkpoint):
    payload = torch.load(checkpoint, weights_only=True)
    payload["trained"] = True
    payload["max_length"] = 2
    payload["state_dict"]["classifier.weight"].zero_()
    payload["state_dict"]["classifier.bias"] = torch.tensor([-5.0, 5.0])
    torch.save(payload, checkpoint)
    result = predict_records(
        [{"source": "example.cpp", "function": "example", "tokens": ["METHOD", "unseen", "END"]}], checkpoint
    )
    prediction = result["predictions"][0]
    assert result["model_status"] == "trained"
    assert prediction["prediction"] == "vulnerable"
    assert prediction["scores"]["vulnerable"] > 0.99
    assert prediction["truncated"] is True
    assert prediction["used_token_count"] == 2
    assert prediction["unknown_token_count"] == 1


def test_padding_does_not_change_the_function_representation(checkpoint):
    model, payload = load_checkpoint(checkpoint)
    method, end = payload["vocab"]["METHOD"], payload["vocab"]["END"]
    with torch.inference_mode():
        alone = model(torch.tensor([[method, end]]), torch.tensor([2]))[0]
        batched = model(torch.tensor([[method, end, 0, 0], [method, method, end, end]]), torch.tensor([2, 4]))[0]
    torch.testing.assert_close(alone, batched)


@pytest.mark.parametrize("field,value", [("representation", "different-ast"), ("trained", None)])
def test_incompatible_checkpoint_is_rejected(checkpoint, field, value):
    payload = torch.load(checkpoint, weights_only=True)
    payload[field] = value
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError):
        load_checkpoint(checkpoint)


def test_invalid_vocabulary_is_rejected(checkpoint):
    payload = torch.load(checkpoint, weights_only=True)
    payload["vocab"][UNK] = 0
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="vocabulary"):
        load_checkpoint(checkpoint)


def test_empty_ast_is_rejected(checkpoint):
    with pytest.raises(ValueError, match="nonempty"):
        predict_records([{"source": "empty.c", "function": "empty", "tokens": []}], checkpoint)
