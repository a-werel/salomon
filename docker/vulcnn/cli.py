"""Host Docker launcher and container-side commands for the VulCNN pipeline."""

import argparse
import contextlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from metadata import STAGES
from helper_functions import describe_functions, replace_comments_with_whitespaces, canonicalize_pdg, run_docker_command

try:
    from AbstractCli import AbstractCli
except ModuleNotFoundError:
    from docker.AbstractCli import AbstractCli


class VulCNNCli(AbstractCli):
    """Run the individual VulCNN pipeline stages."""

    def inspect(self, run: Path) -> dict:
        filename = self.read_json(run / "source.json")["filename"]
        self.run_tool("joern-parse", str(run / "original" / filename), "--language", "c", "--output", str(run / "original.cpg"))
        functions = describe_functions(run / "original.cpg", run / "original-functions")
        self.write_json(run / "functions.json", functions)
        return {"functions": functions}

    def normalize(self, run: Path) -> dict:
        from clean_gadget import clean_gadget
        filename = self.read_json(run / "source.json")["filename"]
        original = (run / "original" / filename).read_text(encoding="utf-8")
        stripped = replace_comments_with_whitespaces(original)
        normalized = "".join(clean_gadget(stripped.splitlines(keepends=True)))
        source = run / "source"
        source.mkdir(exist_ok=False)
        (source / filename).write_text(normalized, encoding="utf-8")
        return {"source": str(source / filename)}


    def parse(self, run: Path) -> dict:
        filename = self.read_json(run / "source.json")["filename"]
        cpg = run / "cpg.bin"
        self.run_tool("joern-parse", str(run / "source" / filename), "--language", "c", "--output", str(cpg))
        return {"cpg": str(cpg)}


    def export(self, run: Path) -> dict:
        original = self.read_json(run / "functions.json")
        normalized = describe_functions(run / "cpg.bin", run / "pdgs", export=True)
        if len(original) != len(normalized):
            raise ValueError("Unable to safely map normalized functions to original source functions.")

        for before, after in zip(original, normalized):
            if before.get("line") != after.get("line"):
                raise ValueError("Unable to safely map normalized functions to original source functions.")

        functions = []
        for before, after in zip(original, normalized):
            canonicalize_pdg(Path(after["pdg"]))
            functions.append({**before, "normalized_name": after["name"], "pdg": after["pdg"]})
        manifest = {"schema_version": 1, "filename": self.read_json(run / "source.json")["filename"], "functions": functions}
        self.write_json(run / "result.json", manifest)
        return manifest


    def embed(self, run: Path, model_path: Path) -> dict:
        import ImageGeneration
        import numpy as np
        import sent2vec

        manifest = self.read_json(run / "result.json")

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
        self.write_json(run / "result.json", manifest)
        return manifest


    def classify(self, run: Path, nn_model: Path, max_length: int) -> dict:
        import numpy as np
        import torch
        from model import TextCNN

        # Read the manifest file.
        manifest = self.read_json(run / "result.json")

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
                padded_feature_tensor = np.zeros((3, max_length, manifest["embedding_dim"]), dtype=np.float32)
                # Pad the channels to the max_length.
                length = min(channels.shape[1], max_length)
                padded_feature_tensor[:, :length, :] = channels[:, :length, :]

                logits, _ = model(torch.from_numpy(padded_feature_tensor).unsqueeze(0))
                label = int(logits.argmax(dim=1).item())
                score = float(logits.softmax(dim=1)[0, 1].item())

                predictions.append((padded_feature_tensor, label))
                rows.append({**function, "label": label, "vulnerable_score": score})

        with (run / "predictions.pkl").open("wb") as stream:
            pickle.dump(predictions, stream)

        self.write_json(run / "predictions.json", rows)

        vulnerable = [row for row in rows if row["label"] == 1]
        report = run / "report.txt"
        lines = [
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

    def run_stage(self, args) -> dict:
        with contextlib.redirect_stdout(sys.stderr):
            if args.command == "inspect":
                return self.inspect(args.run)
            elif args.command == "normalize":
                return self.normalize(args.run)
            elif args.command == "parse":
                return self.parse(args.run)
            elif args.command == "export":
                return self.export(args.run)
            elif args.command == "embed":
                return self.embed(args.run, args.model)
            elif args.command == "classify":
                return self.classify(args.run, args.nn_model, args.max_length)
            raise ValueError(f"Unknown stage: {args.command}")

def main() -> int:
    parser = argparse.ArgumentParser(description="VulCNN Docker launcher and pipeline stages.")
    commands = parser.add_subparsers(dest="command", required=True)

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
            result = VulCNNCli().run_stage(args)
                    
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
