"""Host Docker launcher and container-side commands for the VulCNN pipeline."""

import argparse
import contextlib
import html
import json
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

""" 
Define the stages of the VulCNN pipeline:
1. Inspect: Inspect the source code and generate a CPG.
2. Normalize: Normalize the source code and generate a normalized CPG. (remove commands, normalize identifiers)
3. Parse: Parse the normalized CPG and generate a PDG.
4. Export: Export the PDG as JSON files.
5. Embed: Embed the PDG into a feature vector.
6. Classify: Classify the feature vector into either vulnerable or non-vulnerable.
"""

STAGES = ("inspect", "normalize", "parse", "export", "embed", "classify")

"""
Define the upstream scripts that are used in the VulCNN pipeline:
- normalization.py: Normalize the source code and generate a normalized CPG. (remove commands, normalize identifiers)
- ImageGeneration.py: Embed the PDG into a feature vector.
- generate_train_test_data.py: Generate the training and test data for the VulCNN model.
- VulCNN.py: Classify the feature vector into either vulnerable or non-vulnerable.
"""
UPSTREAM_SCRIPTS = ("normalization.py", "ImageGeneration.py", "generate_train_test_data.py", "VulCNN.py")


def read_json(path: Path):
    """Read a JSON file and return the parsed JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    """Write a JSON object to a file."""
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def tool(*arguments, cwd: Path | None = None) -> None:
    """Run a tool with the given arguments and working directory."""
    subprocess.run(list(arguments), check=True, stdout=sys.stderr, cwd=cwd)


def prepare_docker_wrapper(args, job_name, workspace, models) -> list:
    command = ["docker", "run", "--rm", "--init", "--name", job_name]
        # Add the mount points for the workspace and models directories.
    command += [
        "--mount",
        f"type=bind,src={workspace},dst=/workspace",
        "--mount",
        f"type=bind,src={models},dst=/models,readonly",
        "--entrypoint",
        "timeout",
        args.image,
    ]
    return command

def prepare_docker_timeout_run(command, args):
    command += [
        "--signal=TERM",
        "--kill-after=5",
        str(args.timeout),
        "python",
        "/opt/salomon-vulcnn/cli.py",
        args.stage,
        *args.parameters,
    ]
    return command

def run_docker_command(args) -> dict:
    """All Docker calls live here. Arguments are passed directly without a shell."""
    
    # Resolve the workspace and models directories.
    workspace = args.workspace.resolve()
    models = args.models.resolve()
    
    # Generate a unique job name for the container. This is used to identify the container in the docker system.
    job_name = f"salomon-vulcnn-{uuid4().hex}"
    # If a container is provided, use it, otherwise run a new container.
    
    if args.container:
        # Execute the command in the existing container.
        command = ["docker", "exec", args.container, "timeout"]
    else:
        
        # Run a new container.
        # We need to build a wrapper around the command to handle the timeout and the container cleanup.
        command = prepare_docker_wrapper(args, job_name, workspace, models)
    
    # Add the timeout command to the command list.
    command = prepare_docker_timeout_run(command, args)
    
    # Print the command to the console.
    print(f"Running command: {' '.join(command)}", file=sys.stderr)
    # Try to run the command.
    # Run the command and capture the output.

    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=args.timeout + 10)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    return json.loads(completed.stdout)


def describe_functions(cpg: Path, destination: Path, export: bool = False) -> list[dict]:
    # Read the functions.sc script.
    # Functions.sc is a JOERN script that is used to extract the functions from the CPG.
    script = Path(__file__).with_name("functions.sc")
    # Run the JOERN script to extract the functions from the CPG and save the results to the destination directory.
    tool(
        "joern",
        "--script",
        str(script),
        "--params",
        f"cpgFile={cpg},outputPath={destination},exportPdg={str(export).lower()}",
        cwd=cpg.parent,
    )
    functions = read_json(destination / "functions.json")

    if not functions:
        raise ValueError("No internal function definitions were found in the source code.")

    return functions


def inspect_source(run: Path) -> dict:
    filename = read_json(run / "source.json")["filename"]
    tool("joern-parse", str(run / "original" / filename), "--language", "c", "--output", str(run / "original.cpg"))
    functions = describe_functions(run / "original.cpg", run / "original-functions")
    write_json(run / "functions.json", functions)
    return {"functions": functions}

def replace_comments_with_whitespaces(original):
    return re.sub(r"(?<!:)//.*|/\*(\s|.)*?\*/", lambda match: re.sub(r"[^\r\n]", " ", match.group()), original)

def normalize_source(run: Path) -> dict:
    from clean_gadget import clean_gadget

    
    # Read the filename from the source.json file.
    filename = read_json(run / "source.json")["filename"]
    # Read the original source code from the original directory.
    original = (run / "original" / filename).read_text(encoding="utf-8")
    
    # Remove the comments and normalize the identifiers.
    # Upstream comment removal and identifier normalization, retaining blank lines
    # so original and normalized functions can be matched by source position.
    stripped = replace_comments_with_whitespaces(original)
    normalized = "".join(clean_gadget(stripped.splitlines(keepends=True)))

    source = run / "source"
    source.mkdir(exist_ok=False)
    (source / filename).write_text(normalized, encoding="utf-8")
    return {"source": str(source / filename)}


def parse_source(run: Path) -> dict:
    filename = read_json(run / "source.json")["filename"]
    cpg = run / "cpg.bin"
    tool("joern-parse", str(run / "source" / filename), "--language", "c", "--output", str(cpg))
    return {"cpg": str(cpg)}


def export_pdgs(run: Path) -> dict:
    original = read_json(run / "functions.json")
    normalized = describe_functions(run / "cpg.bin", run / "pdgs", export=True)
    functions = []
    for before, after in zip(original, normalized, strict=True):
        canonicalize_pdg(Path(after["pdg"]))
        functions.append({**before, "normalized_name": after["name"], "pdg": after["pdg"]})
    manifest = {"schema_version": 1, "filename": read_json(run / "source.json")["filename"], "functions": functions}
    write_json(run / "result.json", manifest)
    return manifest


def canonicalize_pdg(path: Path) -> None:
    """Convert Joern HTML labels to the quoted labels used in VulCNN's PDG corpus."""

    def quote_label(match):
        label = re.sub(r"<SUB>.*?</SUB>", "", match.group(1), flags=re.DOTALL)
        label = re.sub(r"<BR\s*/?>", "\n", label, flags=re.IGNORECASE)
        return "label = " + json.dumps(html.unescape(label))

    dot = path.read_text(encoding="utf-8")
    dot = re.sub(r"label\s*=\s*<(.*?)>(?=\s*\])", quote_label, dot, flags=re.DOTALL)
    path.write_text(dot, encoding="utf-8")


def embed_pdgs(run: Path, model_path: Path) -> dict:
    import ImageGeneration
    import numpy as np
    import sent2vec

    manifest = read_json(run / "result.json")

    model = sent2vec.Sent2vecModel()
    model.load_model(str(model_path))
    ImageGeneration.sent2vec_model = model

    features = run / "features"
    features.mkdir(exist_ok=False)

    for function in manifest["functions"]:
        channels = ImageGeneration.image_generation(function["pdg"])
        array = np.asarray(channels, dtype=np.float32)
        path = features / f"{function['id']}.npy"
        np.save(path, array, allow_pickle=False)
        function["features"] = str(path)

    manifest["embedding_dim"] = model.get_emb_size()
    write_json(run / "result.json", manifest)
    return manifest


def classify_functions(run: Path, nn_model: Path, max_length: int) -> dict:
    import numpy as np
    import torch
    from model import TextCNN

    # Read the manifest file.
    manifest = read_json(run / "result.json")

    # Create the model.
    model = TextCNN(hidden_size=manifest["embedding_dim"])
    weights = torch.load(nn_model, map_location="cpu", weights_only=True)
    model.load_state_dict(weights.get("state_dict", weights), strict=True)
    model.eval()

    predictions = []
    rows = []

    with torch.inference_mode():
        for function in manifest["functions"]:

            # channels is a 3D numpy array of shape (3, max_length, embedding_dim).
            # 3 is the number of channels in the image.
            # max_length is the maximum length of the sequence.
            # embedding_dim is the dimension of the embedding.
            # function["features"] is the path to the feature vector.
            # Features in this context are the image representations of the functions.
            
            channels = np.load(function["features"], allow_pickle=False)
            tensor = np.zeros((3, max_length, manifest["embedding_dim"]), dtype=np.float32)
            # Pad the channels to the max_length.
            length = min(channels.shape[1], max_length)
            tensor[:, :length, :] = channels[:, :length, :]

            logits, _ = model(torch.from_numpy(tensor).unsqueeze(0))
            label = int(logits.argmax(dim=1).item())
            score = float(logits.softmax(dim=1)[0, 1].item())

            predictions.append((tensor, label))
            rows.append({**function, "label": label, "vulnerable_score": score})

    with (run / "predictions.pkl").open("wb") as stream:
        pickle.dump(predictions, stream)

    write_json(run / "predictions.json", rows)

    vulnerable = [row for row in rows if row["label"] == 1]
    # Create the report file.
    report = run / "report.txt"
    # Create the lines for the report.
    lines = [
        # The title of the report.
        "VulCNN classification report",
        f"Source: {manifest['filename']}",
        f"Analyzed functions: {len(rows)}",
        f"Predicted vulnerable: {len(vulnerable)}",
        "",
        "Vulnerable functions:",
    ]

    lines.extend(f"- {row['name']} (line {row['line']}; {row['full_name']})" for row in vulnerable)
    
    if not vulnerable:
        lines.append("None predicted.")
    
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"report": str(report), "predictions": str(run / "predictions.pkl"), "functions": rows}


def main() -> int:
    # Parse the arguments for the cli.py script.
    parser = argparse.ArgumentParser(description="VulCNN Docker launcher and pipeline stages.")
    # Add the subparsers for the commands. Subparsers are used to add subcommands to the main parser.
    # dest is the name of the attribute that will be used to store the subcommand.
    commands = parser.add_subparsers(dest="command", required=True)
    # Add the parser for the docker command.

    docker = commands.add_parser("docker", help="Launch a pipeline stage through Docker.")
    docker.add_argument("--workspace", type=Path, required=True)
    docker.add_argument("--models", type=Path, required=True)
    docker.add_argument("--image", default="salomon-vulcnn:local")
    docker.add_argument("--container")
    docker.add_argument("--timeout", type=int, default=600)
    docker.add_argument("stage", choices=STAGES)
    docker.add_argument("parameters", nargs=argparse.REMAINDER)

    for stage in STAGES:
        command = commands.add_parser(stage)
        command.add_argument("--run", type=Path, required=True)
        if stage == "embed":
            command.add_argument("--model", type=Path, required=True)
        if stage == "classify":
            command.add_argument("--nn-model", "--nn_model", dest="nn_model", type=Path, required=True)
            command.add_argument("--max-length", type=int, default=100)
    
    args = parser.parse_args()

    sys.path.insert(0, os.environ.get("VULCNN_HOME", "/opt/VulCNN"))
    
    try:
        if args.command == "docker":
            result = run_docker_command(args)
        else:
            # Reserve stdout for JSON; upstream functions occasionally print.
            with contextlib.redirect_stdout(sys.stderr):
                if args.command == "embed":
                    result = embed_pdgs(args.run, args.model)
                elif args.command == "classify":
                    result = classify_functions(args.run, args.nn_model, args.max_length)
                else:
                    result = {
                        "inspect": inspect_source,
                        "normalize": normalize_source,
                        "parse": parse_source,
                        "export": export_pdgs,
                    }[args.command](args.run)
                    
    except subprocess.CalledProcessError as error:
        if error.stderr:
            print(error.stderr, file=sys.stderr, end="")
        print(f"VulCNN command failed: {error}", file=sys.stderr)
        return error.returncode if error.returncode > 0 else 1
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"VulCNN command failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
