import json
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from uuid import uuid4


# Replace "Model" everywhere with appropriate model names and variables
class AbstractBackend(ABC):
    def __init__(
        self,
        image: str,
        nn_model: str,
        workspace: str | Path | None = None,
        models: str | Path | None = None,
        container: str | None = None,
        # probably
        max_length: int = 100,
        timeout: int = 600,
    ):
        # TODO
        self.workspace = Path(workspace or os.environ.get("EXAMPLE_WORKSPACE", "outputs/salomon")).resolve()
        self.models = Path(models or os.environ.get("EXAMPLE_MODELS", "models")).resolve()

        self.image = image

        # TODO 
        self.container = container or os.environ.get("EXAMPLE_CONTAINER")
        self.nn_model = self._model_path(nn_model)

        if max_length < 10 or timeout < 0:
            raise ValueError("max_length must be at least 10 and timeout must be positive.")

        self.max_length = max_length
        self.timeout = timeout

        # TODO
        self.cli = Path(__file__).resolve().parents[2] / "cli.py"

        # TODO
        if not self.cli.is_file():
            self.cli = Path(__file__).resolve().parents[4] / "docker/example/cli.py"
    
    @staticmethod
    def _model_path(value: str) -> str:
        path = Path(value)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError("Model path must be relative to model's directory")
        return path.as_posix()
    
    # TODO - im not implementing this because we will use multiple models 
    # and I don't see a reason to keep them inside a list for readability
    @abstractmethod
    def check_models(self) -> None:
        """check whether every required model file is present in the models directory"""

    def prepare_source(self, source_code: str | bytes, filename: str):
        
        if isinstance(source_code, bytes):
            source_code = source_code.decode("utf-8", errors="strict")
        if not isinstance(source_code, str) or not source_code.strip():
            raise ValueError("Source code must contain UTF-8 C/C++ source code.")
        if Path(filename).name != filename or '\\' in filename or '//' in filename:
            raise ValueError("Filename must be a filename without directory components.")
        if Path(filename).suffix.lower() not in ('.c', '.cc', '.cpp', '.cxx', '.hpp', '.h'):
            raise ValueError("Filename must have a C/C++ extension.")

        run = self.workspace / uuid4().hex
        original = run / "original"
        original.mkdir(parents=True)
        (original / filename).write_text(source_code, encoding="UTF-8")
        (run / "source.json").write_text(json.dumps({"filename": filename}), encoding="UTF-8")
        return run
    
    def container_path(self, path: Path) -> str:
        return str(Path("workspace") / path.resolve().relative_to(self.workspace))
    
    def host_path(self, path: str) -> Path:
        relative = Path(path).relative_to("/workspace")
        result = (self.workspace / relative).resolve()
        if not result.is_relative_to(self.workspace):
            raise ValueError(f"{path} escaped the workspace.")
        return result

    #TODO
    @abstractmethod
    def create_model_representation(self, run: Path) -> dict:
        """return a manifest result manifest file as a dict"""

    def run_model_command(self, command, **parameters) -> dict:
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
            self.image,
            "--timeout",
            str(self.timeout)
        ]

        if self.container:
            arguments.extend(["--container", self.container])
        
        arguments.append(command)

        for key, value in parameters.items():
            arguments.extend([f"--{key.replace("_", "-")}", str(value)])
        
        completed = subprocess.run(arguments, check=True, capture_output=True, text=True, timeout = self.timeout + 30)
        return json.loads(completed.stdout)
