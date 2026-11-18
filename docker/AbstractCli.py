from abc import ABC
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4


class AbstractCli(ABC):
    def __init__(
        self,
        cli_path: Path | None = None,
        models: Path | None = None,
        workspace: Path | None = None,
        container: str | None = None,
        timeout: int = 600,
    ):
        self.models = models.resolve() if models is not None else None
        self.workspace = workspace.resolve() if workspace is not None else None
        self.container = container
        self.timeout = timeout
        self.cli_path = cli_path.resolve() if cli_path is not None else None

        if self.timeout < 0:
            raise ValueError("Timeout can't be lower than 0.")

    @staticmethod
    def read_json(path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def write_json(path: Path, value) -> None:
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")

    @staticmethod
    def run_tool(arguments: list[str], cwd: Path | None = None) -> None:
        # stdout is used by the json protocol, so we have to redirect stdout to stderr
        # for logging output
        subprocess.run(arguments, check=True, stdout=sys.stderr, cwd=cwd)

    @staticmethod
    def prepare_docker_command(image: str, job_name: str, workspace: Path, models: Path) -> list[str]:
        command = ["docker", "run", "--rm", "--init", "--name", job_name]

        command += [
            "--mount",
            f"type=bind,src={workspace},dst=/workspace",
            "--mount",
            f"type=bind,src={models},dst=/models,readonly",
            "--entrypoint",
            "timeout",
            image,
        ]

        return command

    def prepare_docker_timeout_run(self, command: list[str], stage: str, parameters: list[str]) -> list[str]:
        command += [
            "--signal=TERM",
            "--kill-after=5",
            str(self.timeout),
            "python",
            self.cli_path.as_posix(),
            stage,
            *parameters
        ]

        return command

    def build_docker_command(self, image: str, stage: str, parameters: list[str]) -> tuple[list[str], str]:
        job_name = f"salomon-vulcnn-{uuid4().hex}"

        if self.container:
            command = ["docker", "exec", self.container, "timeout"]
        else:
            command = self.prepare_docker_command(image, job_name, self.workspace, self.models)

        return self.prepare_docker_timeout_run(command, stage, parameters), job_name

    def run_docker_command(self, image: str, stage: str, parameters: list[str]) -> dict:
        command, job_name = self.build_docker_command(image, stage, parameters)
        print(f"Running command: {' '.join(command)}", file=sys.stderr)

        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout + 10
            )
        except subprocess.TimeoutExpired:
            if not self.container:
                subprocess.run(
                    ["docker", "rm", "--force", job_name],
                    check=False,
                    capture_output=True,
                    text=True
                )
            raise

        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        # json protocol
        return json.loads(completed.stdout)
        