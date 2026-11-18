import re
from pathlib import Path
import json
import html

try:
    from AbstractCli import AbstractCli
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from AbstractCli import AbstractCli

def check_if_functions_are_empty(destination: Path) -> list[dict]:
    functions = AbstractCli.read_json(destination / "functions.json")
    if not functions:
        raise ValueError("No internal function definitions were found in the source code.")
    return functions

def describe_functions(cpg: Path, destination: Path, export: bool = False) -> list[dict]:
    script = Path(__file__).with_name("functions.sc")
    AbstractCli.run_tool(
        [
            "joern",
            "--script",
            str(script),
            "--params",
            f"cpgFile={cpg},outputPath={destination},exportPdg={str(export).lower()}",
        ],
        cwd=cpg.parent,
    )
    # check whether the output is not empty

    return check_if_functions_are_empty(destination)


COMMENT_PATTERN = re.compile(
    r"""
    (?<!:)//[^\r\n]*   # single-line comment, except URL-like ://
    |                  # or
    /\*[\s\S]*?\*/     # multi-line comment
    """,
    re.VERBOSE,
)
SUB_TAG_PATTERN = re.compile(r"<SUB>.*?</SUB>", re.DOTALL)
BR_TAG_PATTERN = re.compile(r"<BR\s*/?>", re.IGNORECASE)
HTML_LABEL_PATTERN = re.compile(
    r"""
    label\s*=\s*<(?P<label>.*?)>
    (?=\s*\])
    """,
    re.DOTALL | re.VERBOSE,
)


def blank_comment(match: re.Match[str]) -> str:
    return "".join(char if char in "\r\n" else " " for char in match.group())


def replace_comments_with_whitespaces(original):
    return COMMENT_PATTERN.sub(blank_comment, original)


def quote_pdg_label(match: re.Match[str]) -> str:
    # remove labels
    label = SUB_TAG_PATTERN.sub("", match.group("label"))
    # convert brs into real newlines
    label = BR_TAG_PATTERN.sub("\n", label)
    return "label = " + json.dumps(html.unescape(label))


def canonicalize_pdg(path: Path) -> None:
    """Convert Joern HTML labels to the quoted labels used in VulCNN's PDG corpus."""
    dot = path.read_text(encoding="utf-8")
    dot = HTML_LABEL_PATTERN.sub(quote_pdg_label, dot)
    path.write_text(dot, encoding="utf-8")

def run_docker_command(args) -> dict:
    docker_cli = AbstractCli(
        cli_path=Path(__file__).with_name("cli.py"),
        models=args.models,
        workspace=args.workspace,
        container=args.container,
        timeout=args.timeout,
    )
    return docker_cli.run_docker_command(args.image, args.stage, args.parameters)