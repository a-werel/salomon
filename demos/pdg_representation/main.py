import os
import subprocess
from pathlib import Path

import questionary
from graphviz import Source

path_to_example = Path("/home/demo/examples/")
path_to_output = Path("/home/demo/output/")

for file in path_to_output.iterdir():
    file.unlink()

EXAMPLES = [f for f in os.listdir(path_to_example) if os.path.isfile(path_to_example / f)]

selected = questionary.select("Choose file:", choices=EXAMPLES).ask()

path_to_example = path_to_example / selected

print(path_to_example)
p = subprocess.Popen(["joern", "--script", "/home/demo/app/export_pdg.sc", "--param", f"inputPath={path_to_example}"])
print("Parsing...")
return_code = p.wait()
print(f"Finished with code {return_code}.")

if return_code != 0:
    raise SystemExit(return_code)

for dot_path in path_to_output.glob("*.dot"):
    graph = Source.from_file(dot_path)
    graph.render(format="svg")
