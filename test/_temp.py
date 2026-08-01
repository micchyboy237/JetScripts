import json
import shutil
from pathlib import Path

from jet.file.utils import load_file
from rich.console import Console
from wtpsplit import SaT

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# sat = SaT("sat-12l-sm")
sat = SaT("sat-3l-sm")
# optionally run on GPU for better performance
# also supports TPUs via e.g. sat.to("xla:0"), in that case pass `pad_last_batch=True` to sat.split
# sat.half().to("cuda")

# Example 1
data = load_file(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/utils/generated/run_extract_headers/headings.json"
)
text = data[0]["content"]
results_1 = sat.split(text)
console.print(f"\nResults 1 ({len(results_1)})")

# Example 2
data = load_file(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/utils/generated/run_extract_headers/headings_with_links.json"
)
text = data[0]["content"]
results_2 = sat.split(text)
console.print(f"\nResults 2 ({len(results_2)})")

output_path_json_1 = OUTPUT_DIR / "sentences.json"
output_path_json_2 = OUTPUT_DIR / "sentences_with_links.json"

# Save both files
with open(output_path_json_1, "w") as f:
    json.dump(results_1, f, indent=2)

with open(output_path_json_2, "w") as f:
    json.dump(results_2, f, indent=2)

# Display saved paths with resource links
console.print("\n[bold green]✓ Processing complete![/bold green]")
console.print(f"[cyan]Output files:[/cyan]")
console.print(f"  • [link=file://{output_path_json_1}]{output_path_json_1.name}[/link]")
console.print(f"  • [link=file://{output_path_json_2}]{output_path_json_2.name}[/link]")
