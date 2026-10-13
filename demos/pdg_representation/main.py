import os
import subprocess
from pathlib import Path

import questionary

path_to_example = Path("/home/demo/examples/")
EXAMPLES = [f for f in os.listdir(path_to_example) if os.path.isfile(path_to_example / f)]

selected = questionary.select("Choose file:", choices=EXAMPLES).ask()

path_to_example = path_to_example / selected

print(path_to_example)
p = subprocess.Popen(["joern", "--script", "/home/demo/app/export_pdg.sc", f"--param inputPath={path_to_example}"])
print("Parsing...")
return_code = p.wait()
print(f"Finished with code {return_code}.")
