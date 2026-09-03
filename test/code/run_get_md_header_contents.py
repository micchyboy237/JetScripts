import json
import shutil
from pathlib import Path

from jet.code.splitter_markdown_utils import get_md_header_contents
from rich.console import Console
from rich.table import Table

console = Console()
SCRIPT_NAME = Path(__file__).stem
OUTPUT_DIR = Path(__file__).parent / "generated" / SCRIPT_NAME
DATA_PATH = Path(__file__).parent / "data" / "sample1.md"

# Setup output directory
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    console.rule(f"[bold blue]{SCRIPT_NAME}")
    console.log(f"📂 Input: {DATA_PATH.name}")

    if not DATA_PATH.exists():
        console.log(f"[red]❌ Data file not found: {DATA_PATH}")
        return

    try:
        console.log("⚙️  Running get_md_header_contents...")
        md_text = DATA_PATH.read_text(encoding="utf-8")
        headers = get_md_header_contents(md_text)

        output_file = OUTPUT_DIR / f"{DATA_PATH.stem}_contents.json"
        serializable = [{"text": h.text, "metadata": dict(h.metadata)} for h in headers]
        output_file.write_text(
            json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Summary table
        table = Table(title=f"Extracted Headers ({len(headers)})")
        table.add_column("Level", style="cyan")
        table.add_column("Header", style="green")
        table.add_column("Tokens", justify="right")
        for h in headers[:10]:  # Show first 10
            table.add_row(
                str(h.metadata["header_level"]),
                h.metadata["header"][:40],
                str(len(h.text)),
            )
        console.print(table)

        console.log(f"[green]✅ Saved: {output_file.relative_to(Path.cwd())}")

        # Required resource link display
        console.print(
            f"\n🔗 Resource: [link=file://{output_file.absolute()}]{output_file.name}[/link]"
        )

    except Exception as e:
        console.log(f"[red]❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
