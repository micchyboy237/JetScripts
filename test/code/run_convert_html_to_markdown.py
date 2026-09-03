import shutil
from pathlib import Path

from jet.code.markdown_utils import convert_html_to_markdown
from rich.console import Console
from rich.panel import Panel

console = Console()
SCRIPT_NAME = Path(__file__).stem
OUTPUT_DIR = Path(__file__).parent / "generated" / SCRIPT_NAME
DATA_PATH = Path(__file__).parent / "data" / "sample1.html"

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
        console.log("⚙️  Running convert_html_to_markdown (enhanced)...")
        md_content = convert_html_to_markdown(DATA_PATH)

        output_file = OUTPUT_DIR / f"{DATA_PATH.stem}_converted.md"
        output_file.write_text(md_content, encoding="utf-8")

        console.log(f"[green]✅ Saved: {output_file.relative_to(Path.cwd())}")
        console.print(
            Panel(md_content[:500] + "...", title="Preview", border_style="dim")
        )

        # Required resource link display
        console.print(
            f"\n🔗 Resource: [link=file://{output_file.absolute()}]{output_file.name}[/link]"
        )

    except Exception as e:
        console.log(f"[red]❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
