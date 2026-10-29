"""Host-side adapter; PyTorch and Joern stay inside Docker."""

import csv
import io
import json
import subprocess
from pathlib import Path


def bind_mount(source: Path, target: str) -> str:
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="").writerow(["type=bind", f"source={source}", f"target={target}", "readonly"])
    return buffer.getvalue()


class LSTMAdapter:
    def __init__(self, image: str = "salomon-lstm:latest", checkpoint: str | Path | None = None, timeout: int = 600):
        self.image = image
        self.checkpoint = Path(checkpoint).resolve() if checkpoint is not None else None
        if timeout < 1:
            raise ValueError("Timeout must be positive.")
        self.timeout = timeout

    def analyze_code(self, file_path: str | Path) -> dict:
        source = Path(file_path).resolve(strict=True)
        if self.checkpoint is not None and not self.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")
        command = ["docker", "run", "--rm", "--mount", bind_mount(source.parent, "/input")]
        if self.checkpoint is not None:
            command += ["--mount", bind_mount(self.checkpoint, "/model/model.pt")]
        command += [self.image, "predict", "--input", f"/input/{source.name}", "--timeout", str(self.timeout)]
        if self.checkpoint is not None:
            command += ["--checkpoint", "/model/model.pt"]
        try:
            process = subprocess.run(command, capture_output=True, text=True, check=True, timeout=self.timeout + 30)
        except FileNotFoundError as error:
            raise RuntimeError("Docker CLI is not installed or is not on PATH.") from error
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"LSTM container failed: {error.stderr.strip()}") from error
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("LSTM container timed out.") from error
        try:
            result = json.loads(process.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError("LSTM container did not return valid JSON.") from error
        if not isinstance(result, dict) or not isinstance(result.get("predictions"), list):
            raise TypeError("LSTM container returned an invalid response.")
        for prediction in result["predictions"]:
            if not isinstance(prediction, dict) or not isinstance(prediction.get("source"), str):
                raise TypeError("LSTM container returned an invalid function prediction.")
            container_source = prediction.get("source", "")
            if container_source.startswith("/input/"):
                prediction["source"] = str(source.parent / container_source.removeprefix("/input/"))
            elif not Path(container_source).is_absolute():
                base = source if source.is_dir() else source.parent
                prediction["source"] = str(base / container_source)
        return result
