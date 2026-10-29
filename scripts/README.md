# Separate LSTM training script

`train_lstm.py` trains the existing `AstLSTM` from `src/salomon/adapters/lstm/model.py`.
It does not add training to the inference worker or Docker startup.

From the repository root, with Docker Desktop running and `salomon-lstm:latest` built:

```powershell
uv run --extra lstm python scripts/train_lstm.py --input scripts/examples/lstm_training --labels scripts/examples/lstm_training/labels.json --output models/lstm/trained-demo.pt --epochs 20
```

Docker exports the AST once. Training then runs locally on CPU. The script writes:

- `trained-demo.pt`: best checkpoint by validation loss, including trained weights,
  the training vocabulary, architecture and `trained: true`.
- `trained-demo.metrics.json`: selected epoch, loss, accuracy, precision, recall,
  F1, confusion counts and training history.
- `trained-demo.sequences.json`: exported AST records, reusable without Joern.

The six bundled functions are only a runnable pipeline example. They are far too
few to establish vulnerability detection performance. Labels assume valid pointer
arguments and the declared buffer/array sizes. The script never infers labels.

Use the resulting checkpoint with the existing inference image:

```powershell
uv run python -m salomon.adapters.lstm docker/lstm/examples/example.c --checkpoint models/lstm/trained-demo.pt
```

## Your own dataset

### Check learning on 100 functions

After rebuilding the Docker image for `joern-ast-preorder-vars-v2`, prepare new
ASTs and run the diagnostic:

```powershell
uv run --extra lstm python scripts/prepare_vulnllm_sample.py
uv run --extra lstm python scripts/check_lstm_learning.py --dataset data/processed/vulnllm-lstm-vars-v2
```

`check_lstm_learning.py` selects 100 training functions (50 per class), plus 20
held-out validation functions (10 per class), from their existing separate splits.
It uses complete functions of at most 256 tokens, removes duplicate inputs and
excludes inputs with conflicting labels. Existing related-function groups stay
in their original split. The vocabulary uses only the 100 training examples.

The model architecture is unchanged. This diagnostic uses Adam at 0.01, batches
of 16 and up to 100 epochs, stopping when all training examples are classified
correctly. It saves the best checkpoint by **training loss** to
`models/lstm/learning-100-v2.pt`, the exact selection to `.labels.json` and
training history plus final train/validation metrics to `.metrics.json`.
Validation is evaluated once at the end and does not select the checkpoint.

On the first seeded run (42), training reached 100/100 correct at epoch 73;
held-out validation was 10/20 correct. This establishes that the model can fit
the small sample, not that it generalizes. The sample size, sequence selection,
learning rate and epoch count also differ from the earlier 1084-example run,
so this is not an isolated measurement of the tokenization change.

Variable tokens now distinguish declarations and their uses with function-local
IDs such as `IDENTIFIER:VAR_1`. Old AST caches and version-1 weights must be
replaced by freshly exported ASTs and newly trained checkpoints.

### Download a VulnLLM-R sample

```powershell
uv run --extra lstm python scripts/prepare_vulnllm_sample.py
uv run --extra lstm python scripts/train_lstm.py --sequences data/processed/vulnllm-lstm-vars-v2/sequences.json --labels data/processed/vulnllm-lstm-vars-v2/labels.json --output models/lstm/vulnllm-vars-v2.pt --epochs 20
```

The preparation script downloads 2000 functions (1000 per class) from
[VulnLLM-R-Train-Data](https://huggingface.co/datasets/UCSB-SURFI/VulnLLM-R-Train-Data),
using randomly ordered API pages instead of downloading every dataset file.
It selects C/C++ `function_level` rows without multi-function or stack-trace flags,
uses `target` as the binary label, skips truncated API cells and deduplicates code
within each class. Raw rows, dataset revision and seed are saved in `sample.json`.
The official test dataset is not accessed.

Code is saved in `sources/` and parsed by the existing Docker image. Only examples
exporting exactly one nonempty function AST receive labels; exclusions are recorded
in `preparation.json`. Equal function names and identical model inputs are grouped
transitively before an approximately 80/20 train/validation split. This is a
conservative safeguard, not a project-level split: project provenance is often
missing, so other related functions may remain across splits. Reported validation
scores are preliminary, not independent benchmark results.
Preparation uses the same default 512-token limit as training. If changing the
training limit, re-run preparation with matching `--max-length` before training.

Use `--samples 200` for a smaller even-sized sample and choose a different `--output`
directory. `--download-only` saves the code without invoking Docker. Re-running
with the same settings reuses `sample.json` and `sequences.json`.

### Prepare your own labels

Place C/C++ source files in a directory and create a JSON labels manifest:

```json
[
  {"source": "a.cpp", "function": "unsafe_copy", "label": 1, "split": "train", "group": "project-a"},
  {"source": "a.cpp", "function": "safe_copy", "label": 0, "split": "train", "group": "project-a"},
  {"source": "b.cpp", "function": "unsafe_read", "label": 1, "split": "validation", "group": "project-b"},
  {"source": "b.cpp", "function": "safe_read", "label": 0, "split": "validation", "group": "project-b"}
]
```

`label` is `0 = safe`, `1 = vulnerable`. Source names are relative to the input
directory; a label must match exactly one exported function. Only labelled
functions are used. Train and validation are required, each with both classes.
The optional test split is evaluated only after checkpoint selection.

Keep related revisions, safe/vulnerable pairs and projects in one `group` and one
split. If omitted, group defaults to the source name. The script rejects groups
and identical AST sequences shared across splits. The vocabulary is built only
from training functions; unseen validation/test tokens map to `UNK`.

```powershell
uv run --extra lstm python scripts/train_lstm.py --input data/raw/cpp --labels data/raw/labels.json --output models/lstm/trained.pt --epochs 20 --batch-size 32
```

For already exported AST JSON, replace `--input` with `--sequences`:

```powershell
uv run --extra lstm python scripts/train_lstm.py --sequences models/lstm/trained-demo.sequences.json --labels scripts/examples/lstm_training/labels.json --output models/lstm/another-run.pt
```

Sequence records must contain `source`, `function` and nonempty `tokens` arrays
produced by our `joern-ast-preorder-vars-v2` exporter. Arbitrary lexical tokens are not
compatible with the inference pipeline.

Defaults: 20 epochs, batch size 32, sequence limit 512, embedding 32, hidden size
64, Adam learning rate 0.003, seed 42 and one CPU thread. Both training and
evaluation use batches. Existing checkpoint files are not overwritten: choose
a new `--output` for each run. No existing demo weights or inference defaults
are replaced automatically.
