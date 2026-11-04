# Train and export a VulCNN CNN checkpoint

This guide uses your downloaded sent2vec model, the authors' bundled SARD PDGs,
and a patched working copy of their CNN training code. It produces a checkpoint
that Salomon can load at `models/vulcnn/cnn.pt`. It does not retrain sent2vec.

The commands below are for Bash on Linux, run from the Salomon repository root.
Host tools required: Docker with Compose, Git, curl, unzip, sha512sum, and Python 3.
Python ML dependencies run inside the existing VulCNN image, which uses CPU
PyTorch. Training can take substantial time; this guide does not assume a GPU.

## 1. Put the downloaded embedding model in place

```bash
cd /home/riko111/myprojects/salomon
mkdir -p models/vulcnn outputs/vulcnn/training
```

If necessary, extract the download, then copy the actual sent2vec binary to
`models/vulcnn/sent2vec.bin`. Replace the example source path below with yours;
skip the copy if the file is already there.

```bash
cp /path/to/downloaded/model.bin models/vulcnn/sent2vec.bin
test -s models/vulcnn/sent2vec.bin
```

Keep this same embedding file for feature generation and subsequent inference.
Equal embedding dimensions do not make two different embedding models
interchangeable.

## 2. Prepare and build the runtime

```bash
docker info
bash scripts/setup_vulcnn.sh
docker compose build vulcnn
```

`docker info` must succeed. If it reports a Docker socket permission error,
resolve Docker access before proceeding. No Compose service needs to be running
for the training commands below.

Define this helper in your current terminal:

```bash
vulcnn_train() {
    docker run --rm --init \
        --user "$(id -u):$(id -g)" \
        --shm-size=8g \
        --env OMP_NUM_THREADS=4 \
        --env MKL_NUM_THREADS=4 \
        --mount "type=bind,src=$PWD/outputs/vulcnn,dst=/workspace" \
        --mount "type=bind,src=$PWD/models,dst=/models,readonly" \
        --workdir /workspace/training/upstream \
        --entrypoint python \
        salomon-vulcnn:local "$@"
}
```

Each call starts a disposable container. Training outputs persist in
`outputs/vulcnn/training/`. The models mount is read only, so checkpoints are
written to the training workspace and copied into `models/` afterward.
The shared-memory limit accommodates sent2vec's Linux shared-memory loading;
large embedding files also require enough available host memory. Recreate the
helper if you open a new terminal, and keep your working directory at the repo
root.

## 3. Extract the graphs and prepare the authors' scripts

The checkout includes `third_party/VulCNN/pdgs/Vul.zip` and `No-Vul.zip`:

| Directory | Meaning | Label |
| --- | --- | --- |
| `Vul` | Vulnerable SARD functions | `1` |
| `No-Vul` | Non-vulnerable SARD functions | `0` |

These are already exported PDGs. This route skips source normalization and Joern
graph generation for training. New source files analyzed by Salomon still go
through those stages.

Run this preparation block once for a new training directory. It copies four
Python files and extracts the bundled archives. The Git submodule is not edited.
It refuses to overwrite an existing working copy; use a fresh training directory
for another experiment.

```bash
python3 - <<'PY'
from pathlib import Path
from shutil import copy2
from zipfile import ZipFile

upstream = Path("third_party/VulCNN")
training = Path("outputs/vulcnn/training")
working = training / "upstream"
working.mkdir(parents=True, exist_ok=False)
for name in ("ImageGeneration.py", "generate_train_test_data.py", "model.py", "VulCNN.py"):
    copy2(upstream / name, working / name)
for name in ("Vul", "No-Vul"):
    with ZipFile(upstream / "pdgs" / f"{name}.zip") as archive:
        archive.extractall(training / "pdgs")

def replace_once(text, before, after):
    if text.count(before) != 1:
        raise RuntimeError(f"Unexpected upstream code; inspect before patching: {before!r}")
    return text.replace(before, after, 1)

# argparse otherwise leaves the fold count as a string, breaking range().
path = working / "generate_train_test_data.py"
code = path.read_text()
code = replace_once(code, "'-n', '--num',", "'-n', '--num', type=int,")
path.write_text(code)

# Use two feature-generation workers to reduce concurrent graph-processing load.
path = working / "ImageGeneration.py"
code = replace_once(path.read_text(), "Pool(10)", "Pool(2)")
path.write_text(code)

path = working / "model.py"
code = path.read_text()
code = replace_once(
    code,
    "from sklearn.metrics import confusion_matrix",
    "from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, accuracy_score",
)
# precision_recall_fscore_support returns four values, not five. Obtain class
# confusion matrices explicitly, and calculate accuracy without swapping labels.
code = replace_once(
    code,
    "    accuracy = get_accuracy(labels, predictions)\n"
    "    precision, recall, f_score, true_sum, MCM = precision_recall_fscore_support(labels, predictions,average='macro')",
    "    accuracy = accuracy_score(labels, predictions)\n"
    "    MCM = multilabel_confusion_matrix(labels, predictions, labels=[0, 1])",
)
# Clip unscaled gradients before the optimizer step.
code = replace_once(
    code,
    "            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)",
    "            # Gradient clipping is performed before scaler.step().",
)
code = replace_once(
    code,
    "            scaler.scale(loss).backward()",
    "            scaler.scale(loss).backward()\n"
    "            scaler.unscale_(self.optimizer)\n"
    "            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)",
)
# Retain the checkpoint with the lowest validation loss for this fold.
code = replace_once(
    code,
    "        learning_record_dict = {}",
    '        learning_record_dict = {}\n        best_val_loss = float("inf")',
)
code = replace_once(
    code,
    "            val_loss, val_score = self.eval()",
    "            val_loss, val_score = self.eval()\n"
    "            if val_loss < best_val_loss:\n"
    "                best_val_loss = val_loss\n"
    '                checkpoint = os.path.splitext(self.result_save_path)[0] + ".pt"\n'
    '                torch.save({"state_dict": self.model.state_dict(),\n'
    '                            "epoch": epoch + 1, "validation_loss": float(val_loss),\n'
    '                            "max_length": self.max_len, "embedding_dim": self.hidden_size}, checkpoint)\n'
    '                print(f"Saved checkpoint: {checkpoint}")',
)
path.write_text(code)

# Start with one of the five folds and 20 epochs. This is a starting experiment,
# not a claim to reproduce the authors' five-fold, 200-epoch results.
path = working / "VulCNN.py"
code = replace_once(path.read_text(), "range(5)", "range(1)")
code = replace_once(code, "epochs=200", "epochs=20")
path.write_text(code)
print(f"Prepared training scripts in {working}")
PY
```

This fixes the fold argument type, the metrics API error, gradient clipping
order, and missing checkpoint export. It reuses the authors' `TextCNN`, dataset
class, optimizer setup, and training loop. The original scripts remain under
`third_party/VulCNN/`.

## 4. Check the embedding model

```bash
vulcnn_train -c '
import sent2vec
model = sent2vec.Sent2vecModel()
model.load_model("/models/vulcnn/sent2vec.bin")
dimension = model.get_emb_size()
print("Embedding dimension:", dimension)
assert dimension == 128, "Set hidden_size in the working VulCNN.py to this dimension before training."
sent2vec.Sent2vecModel.release_shared_mem("/models/vulcnn/sent2vec.bin")
'
```

The authors' training driver defaults to 128-dimensional embeddings. If yours
has a different dimension, set `hidden_size` in
`outputs/vulcnn/training/upstream/VulCNN.py` accordingly, then continue.
Salomon determines the embedding dimension when it generates inference features.

## 5. Generate feature tensors for both labels

```bash
vulcnn_train ImageGeneration.py \
    -i /workspace/training/pdgs/Vul \
    -o /workspace/training/features/Vul \
    -m /models/vulcnn/sent2vec.bin

vulcnn_train ImageGeneration.py \
    -i /workspace/training/pdgs/No-Vul \
    -o /workspace/training/features/No-Vul \
    -m /models/vulcnn/sent2vec.bin
```

For each graph, the authors' code combines sent2vec embeddings with degree,
closeness, and Katz centrality to generate three feature channels. Output files
appear under `outputs/vulcnn/training/features/{Vul,No-Vul}/`.

Check their counts and shapes:

```bash
vulcnn_train -c '
from pathlib import Path
import pickle
import numpy as np
for label in ("Vul", "No-Vul"):
    graphs = list((Path("/workspace/training/pdgs") / label).glob("*.dot"))
    features = sorted((Path("/workspace/training/features") / label).glob("*.pkl"))
    print(label, "graphs:", len(graphs), "features:", len(features))
    assert len(features) >= 5, "Not enough features for five folds."
    for path in features:
        with path.open("rb") as stream:
            array = np.asarray(pickle.load(stream), dtype=np.float32)
        assert array.ndim == 3 and array.shape[0] == 3 and array.shape[1] > 0, path
        assert np.isfinite(array).all(), path
    print("Last feature shape:", array.shape)
'
```

Upstream feature generation silently skips graphs it cannot process. A smaller
feature count than graph count must be investigated before relying on the
dataset or metrics. Do not continue if an entire class is missing or shape and
finite-value checks fail.

## 6. Assemble the labeled dataset and folds

```bash
vulcnn_train generate_train_test_data.py \
    -i /workspace/training/features \
    -o /workspace/training/pkl/ \
    -n 5
```

Keep the trailing slash on the output path: the upstream script concatenates
filenames with it. It writes `all_data.pkl`, `train.pkl`, and `test.pkl` under
`outputs/vulcnn/training/pkl/`. Each of the five folds has approximately 80% of
each class for training and 20% for evaluation.

The authors' driver uses `test.pkl` during every training epoch; here it is
**validation data used to select the checkpoint**, not an independent test set.
Its random function-level split also does not keep related SARD cases together.
For a trustworthy assessment, reserve an independent test set and group related
cases before splitting. This guide exports a baseline checkpoint; validation
scores alone do not establish real-world vulnerability detection accuracy.

## 7. Train the CNN and save the best checkpoint

```bash
vulcnn_train VulCNN.py -i /workspace/training/pkl/
```

The working driver trains fold 0 for 20 epochs, with the authors' default batch
size of 32 and tensor length of 100. The saved checkpoint is updated whenever
validation loss improves. Training records and weights appear at:

```text
outputs/vulcnn/training/results/
    0_epo20_bat32.result
    0_epo20_bat32.pt
```

To use the authors' longer schedule, change `epochs=20` to `epochs=200` in the
working `VulCNN.py` before training. Change `range(1)` to `range(5)` to train all
five folds separately. Filenames reflect the fold, epoch count, and batch size;
each fold creates a separate model. The commands below assume the initial
one-fold, 20-epoch experiment.

## 8. Install the checkpoint

```bash
cp outputs/vulcnn/training/results/0_epo20_bat32.pt models/vulcnn/cnn.pt
```

This overwrites `models/vulcnn/cnn.pt` if one already exists; retain an existing
checkpoint under a different name if you need it. Keep the training directory
with its metrics and working scripts. Record which embedding file was used:

```bash
sha256sum models/vulcnn/sent2vec.bin models/vulcnn/cnn.pt \
    > outputs/vulcnn/training/model-sha256.txt
```

The resulting layout is:

```text
models/vulcnn/
    sent2vec.bin
    cnn.pt
```

Salomon accepts the checkpoint dictionary's `state_dict` key. Its default
`max_length=100` matches this training configuration.

## 9. Run inference through Salomon

Build the app image if it has not been built, then run the example:

```bash
docker compose build app
mkdir -p data/raw
cp demos/pdg_representation/examples/example1.c data/raw/example.c
docker compose run --rm app analyze /data/raw/example.c
```

The command prints a report path inside `/workspace/<run-id>/`. The same report
is available on your host at `outputs/vulcnn/<run-id>/report.txt`, with
`predictions.json` beside it. Completing this command verifies that the exported
checkpoint works with the analysis pipeline; it does not measure model accuracy.

Stop the Compose runtime when finished:

```bash
docker compose down
```

## Verification status

The preparation block was checked against the pinned upstream source, and the
Python snippets were syntax-checked. Full embedding, training, and inference
must still be run with the downloaded embedding model and a working Docker
daemon. No trained CNN or benchmark results are supplied by this document.
