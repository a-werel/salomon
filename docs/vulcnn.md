# VulCNN integration

## Mandatory local dependencies

**You must initialize the VulCNN checkout and install the Joern release locally
before building the VulCNN image.** They are copied from `third_party`; the
runtime image no longer downloads Joern during its build. The app image can be
built independently with `docker compose build app`.

From the Salomon repository root:

```bash
bash scripts/setup_vulcnn.sh
mkdir -p outputs/vulcnn
docker compose build
```

The setup script initializes the pinned Git submodule at `third_party/VulCNN`
([CGCL-codes/VulCNN](https://github.com/CGCL-codes/VulCNN), commit
`b73810f8c6fcfb12669cfb269eb9aa25b2c67be2`) and downloads Joern **1.1.1125** into
`third_party/joern`. It verifies the release archive against its pinned SHA-512
checksum. Setup needs Git, curl, unzip and sha512sum. Re-running setup reuses the
installed release. Joern is a binary distribution, so it is downloaded rather
than cloned as source; `third_party/joern/` is Git-ignored. VulCNN is tracked as a
submodule. A plain clone of Salomon does not include either dependency's contents.

Joern's local `joern-parse` executable can be invoked with a compatible Java
installation. Docker includes Java 11, Python 3.10, CPU PyTorch, Graphviz and the
bundled native sent2vec implementation, so host Java/Python dependencies are not
needed when using the containers. The Salomon app image uses Python 3.12.

## Run the Python demo on the host

The Python package lives under `src/salomon`. Install it into the environment
that runs the demo so Python can resolve `from salomon...` imports. From the
repository root, using Python 3.12 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python demos/vulcnn_adapter/demo.py
```

Select `.venv/bin/python` as the interpreter if running the demo from an IDE.
For a one-off run without installing the package, use this command from the
repository root:

```bash
PYTHONPATH=src python3 demos/vulcnn_adapter/demo.py
```

The demo calls both `create_pdg` and `analyze_code`, so it requires the built
VulCNN Docker image and the trained model files described below.

## The two Python API methods

```python
from salomon.adapters.vulcnn.adapter import VulCNNAdapter

adapter = VulCNNAdapter()  # outputs/vulcnn and models, relative to your current directory
source = "int add(int a, int b) { return a + b; }\n"

pdgs = adapter.create_pdg(source, filename="example.c")  # list[pathlib.Path]
report = adapter.analyze_code(source.encode("utf-8"), filename="example.c")  # pathlib.Path
print(report.read_text(encoding="utf-8"))
```

Both methods accept a Python string or strictly decoded UTF-8 bytes. The filename
identifies the input and must have a C/C++ extension. Each call creates a separate
run directory and preserves the original source. `create_pdg` needs no model
files. `analyze_code` checks that trained model files exist before starting work.

Supply your trained assets under the Git-ignored `models/vulcnn/` folder:

```text
models/vulcnn/sent2vec.bin
models/vulcnn/cnn.pt
```

The CNN checkpoint must be the upstream `model.TextCNN` state dictionary saved
with `torch.save(model.state_dict(), path)`, or a dictionary containing it under
`state_dict`. Embedding dimensions must match the CNN. The label convention is
**1 = vulnerable, 0 = non-vulnerable**. Default tensor length is 100, as in upstream;
use the padding/truncation length that matches your checkpoint's preprocessing.

Locations and execution settings can be supplied without configuration classes:

```python
adapter = VulCNNAdapter(
    workspace="outputs/vulcnn",
    models="models",
    embedding_model="vulcnn/sent2vec.bin",  # relative to models
    checkpoint="vulcnn/cnn.pt",            # relative to models
    max_length=100,
    timeout=600,                          # seconds per stage
    image="salomon-vulcnn:local",
)
```

The upstream repository supplies no trained CNN checkpoint. The integration tests
create explicitly marked test models to validate the mechanics; those models
have no vulnerability detection accuracy and are never used as default assets.
For commands to train and export a baseline checkpoint using the authors' code,
see [the step-by-step training guide](vulcnn-training.md).

## Call flow and pipeline

```text
adapter.py: create_pdg / analyze_code
    → backend.py: run_vulcnn_command(stage, parameters)
    → docker/vulcnn/cli.py: Docker subprocess
    → same cli.py inside VulCNN: execute the stage
    → shared files and a JSON result
```

`src/salomon/adapters/vulcnn/` contains only `adapter.py` and `backend.py`.
There are no result classes or separate worker module. The backend prepares files
and command arguments; `cli.py` owns Docker execution and container-side stage
dispatch. Arguments are passed as lists without a shell.

Each stage is a separate CLI command routed through `run_vulcnn_command`:

| Command | Work |
| --- | --- |
| `inspect` | Parse the original input and record internal function names, signatures and source positions |
| `normalize` | Remove comments and apply upstream `clean_gadget` identifier normalization to a copy |
| `parse` | Run Joern on the normalized source to create its CPG |
| `export` | Export each internal function's PDG and associate it with its original identity |
| `embed` | Use the supplied sent2vec model and upstream `ImageGeneration.image_generation` to build degree, closeness and Katz channels |
| `classify` | Pad/truncate tensors, load trained CNN weights, infer labels and write predictions and the text report |

`create_pdg` runs the first four commands. `analyze_code` runs all six. Normalization
retains blank lines so original and normalized methods can be paired by line and
source order. A changed method count or line mapping fails the run. Synthetic
methods such as `<global>` and external functions are excluded. Original names
and signatures, rather than normalized `FUN1` names, appear in reports. The small
`functions.sc` Joern script extracts this metadata and the per-function PDGs.
The bridge converts Joern's HTML node labels to the quoted label format in
VulCNN's supplied PDG corpus, removing display-only line subscripts. This makes
the graphs compatible with upstream's DOT reader and node-code extraction.

This implements the analysis path of VulCNN's documented workflow. As confirmed
for this integration, **training is separate**: sent2vec training, labeled dataset
assembly, train/test splitting and CNN training are not repeated for each input.
Existing trained models are used. Upstream's training CLI and remaining scripts
are still available using `upstream <script> <arguments>` inside the runtime.

## Run Salomon and VulCNN together

The root `compose.yaml` includes `docker/app/compose.yaml` and
`docker/vulcnn/compose.yaml`. The VulCNN component owns its build and runtime
settings; the app component declares its dependency on the healthy model runtime.
See [the Docker structure guide](docker.md) for adding other model components.
Running the app starts VulCNN and waits for its health check. The app uses
`docker exec` to launch stage commands in that container.
Compose mounts the local Docker socket into the app for those calls. VulCNN keeps
running between commands; there is no HTTP server or Docker daemon inside it.

```bash
mkdir -p data/raw
cp demos/pdg_representation/examples/example1.c data/raw/example.c
docker build -t salomon-vulcnn:local -f docker/vulcnn/Dockerfile .
docker compose build app
docker compose run --rm app pdg /data/raw/example.c
docker compose run --rm app analyze /data/raw/example.c
docker compose down
```

`analyze` requires the trained assets described above. Running the app without an
operation shows its command help. Use the root Compose setup when running the
app container: a Docker image alone cannot declare a sibling container dependency.
From host Python, the same CLI helper uses a disposable `docker run` for each
stage. Set `VULCNN_CONTAINER=salomon-vulcnn` to reuse the Compose-managed container
from the host, using the same `outputs/vulcnn` and `models` mounts.

| Host | Container | Access |
| --- | --- | --- |
| `outputs/vulcnn/` | `/workspace` | Inputs and generated outputs |
| `models/` | `/models` | Read only |
| `data/` | `/data` in the app | Read only source files |

The host launcher uses your UID/GID for disposable containers. The Compose app
and its `docker exec` commands run as the container user (root by default), so
Compose-generated files normally belong to root on Linux. Per-stage time limits
also execute inside the runtime to stop analysis processes if a client times out.
Failures propagate as subprocess exceptions, with diagnostics on stderr; partial
artifacts remain in their run directory for inspection.

## Outputs and report

```text
outputs/vulcnn/<run-id>/
    original/example.c
    source/example.c
    functions.json          # Original identities
    cpg.bin
    pdgs/<method-id>.dot
    features/<method-id>.npy
    result.json             # Explicit name → PDG → features mapping
    predictions.pkl         # [(numpy tensor, 0 or 1), ...] in source-function order
    predictions.json        # Names, lines, graph paths, labels and model scores
    report.txt
```

Each prediction tensor has shape `(3, max_length, embedding_dimension)`. The model
runs on CPU in evaluation mode without gradients. The text report lists only
functions predicted vulnerable, preserving names and signatures to distinguish
overloads. If no functions are predicted vulnerable, it says so explicitly:

```text
VulCNN classification report
Source: example.c
Analyzed functions: 3
Predicted vulnerable: 2

Vulnerable functions:
- first (line 1; first:int())
- third (line 12; third:int())
```

## Validation

```bash
python -m pytest
RUN_VULCNN_DOCKER_TESTS=1 python -m pytest tests/integration -v
```

Ordinary tests need neither Docker nor trained assets. Opt-in integration tests
use the built image, train a tiny test-only sent2vec fixture, and generate
controlled CNN weights. They exercise real C/C++ graph extraction, normalization,
UTF-8 source preservation, the complete embedding/classification pipeline, mixed
labels `[1, 0, 1]`, tensor shapes, original-name reports and incompatible checkpoints.
They validate integration behavior, not vulnerability detection quality.

The Docker integration tests require access to a running Docker daemon and a
built VulCNN image. A successful Compose configuration check alone does not
validate image builds or model execution.
