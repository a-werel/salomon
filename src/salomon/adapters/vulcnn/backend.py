import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4
from salomon.adapters.AbstractBackend import AbstractBackend


class VulCNNBackend(AbstractBackend):
    def __init__(
        self,
        workspace: str | Path | None = None,
        models: str | Path | None = None,
        image: str = "salomon-vulcnn:local",
        container: str | None = None,
        embedding_model: str = "vulcnn/sent2vec.bin",
        nn_model: str = "vulcnn/cnn.pt",
        max_length: int = 100,
        timeout: int = 600,
    ):
        self.workspace = Path(workspace or os.environ.get("VULCNN_WORKSPACE", "outputs/vulcnn")).resolve()
        self.models = Path(models or os.environ.get("VULCNN_MODELS", "models")).resolve()

        self.image = image
        self.container = container or os.environ.get("VULCNN_CONTAINER")
        self.embedding_model = self._model_path(embedding_model)
        self.nn_model = self._model_path(nn_model)

        if max_length < 10 or timeout <= 0:
            raise ValueError("max_length must be at least 10 and timeout must be positive.")

        self.max_length = max_length
        self.timeout = timeout

        # The wheel includes this same script; source checkouts use docker/vulcnn/cli.py.
        self.cli = Path(__file__).resolve().parents[2] / "vulcnn_cli.py"

        if not self.cli.is_file():
            self.cli = Path(__file__).resolve().parents[4] / "docker/vulcnn/cli.py"

    @staticmethod
    def _model_path(value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Model paths must be relative to the models directory.")
        return path.as_posix()

    def check_models(self) -> None:
        for model in (self.embedding_model, self.nn_model):
            if not (self.models / model).is_file():
                raise FileNotFoundError(f"Required trained model is missing: {self.models / model}")

    def prepare_source(self, source_code: str | bytes, filename: str) -> Path:
        
        if isinstance(source_code, bytes):
            source_code = source_code.decode("utf-8", errors="strict")
        if not isinstance(source_code, str) or not source_code.strip():
            raise ValueError("source_code must contain UTF-8 C/C++ source code.")
        if Path(filename).name != filename or "\\" in filename:
            raise ValueError("filename must be a filename, without directory components.")
        if Path(filename).suffix.lower() not in {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp"}:
            raise ValueError("filename must have a C/C++ extension.")

        run = self.workspace / uuid4().hex
        original = run / "original"
        original.mkdir(parents=True)
        (original / filename).write_text(source_code, encoding="utf-8")
        (run / "source.json").write_text(json.dumps({"filename": filename}), encoding="utf-8")
        return run

    def container_path(self, path: Path) -> str:
        return str(Path("/workspace") / path.resolve().relative_to(self.workspace))

    def host_path(self, path: str) -> Path:
        relative = Path(path).relative_to("/workspace")
        result = (self.workspace / relative).resolve()
        if not result.is_relative_to(self.workspace):
            raise ValueError(f"{path} escaped the workspace.")
        return result

    def create_model_representation(self, run: Path) -> dict:
        for stage in ("inspect", "normalize", "parse", "export"):
            result = self.run_vulcnn_command(stage, run=self.container_path(run))
        return result

    def create_pdg(self, run: Path) -> dict:
        return self.create_model_representation(run)

    def run_vulcnn_command(self, command: str, **parameters) -> dict:
        PYTHON = sys.executable
        CLI_FILE = str(self.cli)

        arguments = [
            PYTHON,
            CLI_FILE,
            "docker",            
            "--workspace",
            str(self.workspace),
            "--models",
            str(self.models),
            "--image",
            self.image,
            "--timeout",
            
            str(self.timeout),
        ]

        # If a container is provided, add it to the arguments.
        if self.container:
            arguments.extend(["--container", self.container])

        arguments.append(command)

        for key, value in parameters.items():
            arguments.extend([f"--{key.replace('_', '-')}", str(value)])
            
        completed = subprocess.run(arguments, check=True, capture_output=True, text=True, timeout=self.timeout + 30)
        return json.loads(completed.stdout)
