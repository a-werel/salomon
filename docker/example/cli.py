import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

# Replace these with the stages and model-specific source files for the new model.
STAGES = ()
UPSTREAM = ()

def prepare_docker_wrapper(args, job_name, workspace, models):
    command = ["docker", "run", "--rm", "--init", "--name", job_name]
    
    command += [
        "--mount",
        f"type=bind,src={workspace},dst=/workspace",
        "--mount",
        f"type=bind,src={models},dst=/models,readonly",
        "--entrypoint",
        "timeout",
        args.image
    ]
    
    return command

def prepare_docker_timeout_run(command, args):
    command += [
        "--signal=TERM",
        "--kill-after=5",
        str(args.timeout),
        "python",
        "/opt/example/cli.py",
        args.stage,
        *args.parameters
    ]
    return command

def run_docker_command(args) -> dict:
    workspace = args.workspace.resolve()
    models = args.models.resolve()
    
    job_name = f"salomon-example-{uuid4().hex}"
    
    if args.container:
        command = ["docker", "exec", args.container, "timeout"]
    else:
        command = prepare_docker_wrapper(args, job_name, workspace, models)
    
    command = prepare_docker_timeout_run(command, args)
    
    print(f"Running command: {' '.join(command)}", file=sys.stderr)
    
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=args.timeout + 10)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    return json.loads(completed.stdout)
        
def main() -> int:
    parser = argparse.ArgumentParser(description="Example Docker launcher and pipeline stages.")
    commands = parser.add_subparsers(dest="command", required=True)
    
    docker = commands.add_parser("docker", help="Launch a pipeline stage through Docker.")
    docker.add_argument("--workspace", type=Path, required=True)
    docker.add_argument("--models", type=Path, required=True)
    docker.add_argument("--image", default="salomon-example:latest")
    docker.add_argument("--container")
    docker.add_argument("--timeout", type=int, default=600)
    docker.add_argument("stage", choices=STAGES)
    docker.add_argument("parameters", nargs=argparse.REMAINDER)
   
    for stage in STAGES:
        command = commands.add_parser(stage)
        command.add_argument("--run", type=Path, required=True)
    
    args = parser.parse_args()
    
    sys.path.insert(0, os.environ.get("EXAMPLE_HOME", "/opt/example"))
    
    result = None
    try:
        if args.command == "docker":
            result = run_docker_command(args)
        else:
            raise NotImplementedError(
                f"Implement the '{args.command}' stage in this example before running it."
            )
    
    except subprocess.CalledProcessError as error:
        if error.stderr:
            print(error.stderr, file=sys.stderr, end="")
        print(f"Example command failed: {error}", file=sys.stderr, end="")
        return error.returncode if error.returncode > 0 else 1
    except (OSError, RuntimeError, ValueError, NotImplementedError, subprocess.TimeoutExpired) as error:
        print(f"Example command failed: {error}", file=sys.stderr, end="")
        return 1
    print(json.dumps(result))
    return 0

def classify_functions(run: Path, nn_model: Path, max_length: int) -> dict:
    raise NotImplementedError
            
if __name__ == "__main__":
    raise SystemExit(main())