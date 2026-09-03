import json
import shutil
from pathlib import Path

from jet.code.splitter_markdown_utils import get_md_header_docs
from rich.console import Console
from rich.panel import Panel

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
        console.log("⚙️  Running get_md_header_docs...")
        md_text = DATA_PATH.read_text(encoding="utf-8")
        docs = get_md_header_docs(md_text, base_url="https://example.com")

        output_file = OUTPUT_DIR / f"{DATA_PATH.stem}_docs.json"
        serializable = [{"text": d.text, "metadata": dict(d.metadata)} for d in docs]
        output_file.write_text(
            json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        console.log(f"[green]✅ Generated {len(docs)} header documents")
        if docs:
            preview = f"First Doc Header: {docs[0].metadata['header']}\nContent Length: {len(docs[0].text)} chars"
            console.print(
                Panel(preview, title="First Document Preview", border_style="dim")
            )

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
