# LSTM inference image

One implementation lives in `src/salomon/adapters/lstm`. The previous
`demos/ast_lstm` training demo has been removed. This image contains Joern,
PyTorch CPU, AST preprocessing and a checkpoint; it does not train at startup.

The bundled checkpoint has reproducible **random, untrained weights**. It tests
integration only. JSON results declare `model_status: "untrained"`,
`assessment_available: false`, and `prediction: null`. The two class scores are
random model outputs, not vulnerability assessments.

## Build and run

Start Docker Desktop with Linux containers. From the repository root:

```powershell
docker build -f docker/lstm/Dockerfile -t salomon-lstm:latest .
uv run python -m salomon.adapters.lstm docker/lstm/examples/example.c
```

The same image can analyze a directory containing C/C++ source files. The host
adapter only requires Docker; PyTorch and Joern run inside the image.

Python integration:

```python
from salomon.adapters.lstm import LSTMAdapter

result = LSTMAdapter().analyze_code("docker/lstm/examples/example.c")
print(result["model_status"])
print(result["predictions"])
```

To call Docker directly in PowerShell:

```powershell
$source = (Resolve-Path docker/lstm/examples).Path
docker run --rm --mount "type=bind,source=$source,target=/input,readonly" salomon-lstm:latest predict --input /input/example.c
```

The image processes `C/C++ -> Joern AST preorder tokens -> embedding -> LSTM ->
class scores`. Joern diagnostics go to stderr; stdout contains a single JSON
response. Function names are metadata, not model features. Unknown tokens and
truncated functions are reported. This is a sequential LSTM, not a Tree-LSTM or
the BGRU from SySeVR.

## Replace the weights later

Supply a compatible checkpoint without changing the adapter:

```powershell
uv run python -m salomon.adapters.lstm docker/lstm/examples/example.c --checkpoint models/lstm/model.pt
```

Or in Python:

```python
adapter = LSTMAdapter(checkpoint="models/lstm/model.pt")
result = adapter.analyze_code("my_project/example.cpp")
```

To bake that checkpoint into an image which needs no mounted model file, use a
small derived Dockerfile in a directory containing `model.pt`:

```dockerfile
FROM salomon-lstm:latest
COPY model.pt /app/checkpoints/model.pt
```

The checkpoint is a `torch.save` dictionary with:

| Key | Required value |
| --- | --- |
| `format_version` | `1` |
| `representation` | `"joern-ast-preorder-v1"` |
| `trained` | `False` for random weights; `True` after actual training |
| `classes` | `["safe", "vulnerable"]`, in this order |
| `architecture` | `{"embedding_dim": 32, "hidden_dim": 64}` or the dimensions used in training |
| `vocab` | Training vocabulary, contiguous integer IDs, `<PAD>=0`, `<UNK>=1` |
| `max_length` | Positive sequence-length limit used in training |
| `state_dict` | Matching `AstLSTM.state_dict()` |

Train the canonical `AstLSTM` on the same AST representation and save its weights
with the vocabulary and metadata above. Arbitrary LSTM/SySeVR/VUDENC weights are
not compatible. Marking a checkpoint trained changes output availability; it does
not establish model accuracy or calibrate its scores.

The image pins Joern to `v4.0.667` and PyTorch to `2.15.0`. Build arguments
`JOERN_VERSION` and `TORCH_VERSION` allow explicit upgrades.

## Checks without Docker

```powershell
uv sync --dev --extra lstm
uv run --extra lstm pytest
```

For a checkpoint/JSON smoke test without starting Joern:

```powershell
uv run --extra lstm python -m salomon.adapters.lstm.worker init-demo --output models/lstm/model.pt
uv run --extra lstm python -m salomon.adapters.lstm.worker predict-sequences --input sequences.json --checkpoint models/lstm/model.pt
```

`sequences.json` must be a nonempty array of records with `source`, `function` and
a nonempty `tokens` array. The Scala exporter next to the worker defines the
checkpoint's AST representation.
