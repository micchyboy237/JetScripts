import shutil
from pathlib import Path

# Rich console for styled resource links
from rich.console import Console

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/code/data/simple_page.html"
