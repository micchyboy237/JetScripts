import argparse
import json
import shutil
from pathlib import Path

from jet.scrapers.html_skeleton_extractor import build_block_tree, extract_skeleton
from rich.console import Console

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_HTML_FILE = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/node_extraction/sample.html"


def main():
    parser = argparse.ArgumentParser(
        description="Extracts heading hierarchies, contents, and metadatas from an HTML file or URL."
    )
    parser.add_argument(
        "html_file",
        nargs="?",
        default=DEFAULT_HTML_FILE,
        help="Path to the HTML file",
    )
    args = parser.parse_args()

    html_file = args.html_file

    with open(html_file, "r", encoding="utf-8") as f:
        html_str = f.read()

    skeleton = extract_skeleton(html_str)
    tree = build_block_tree(html_str, max_words=9999)

    html_file = OUTPUT_DIR / "page.html"
    skeleton_file = OUTPUT_DIR / "skeleton.html"
    tree_file = OUTPUT_DIR / "tree.json"

    html_file.write_text(html_str, encoding="utf-8")
    skeleton_file.write_text(skeleton, encoding="utf-8")
    tree_file.write_text(
        json.dumps(tree, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    console.print(
        f"✅ HTML saved to [bold blue][link=file://{html_file}]{html_file.name}[/link][/bold blue]"
    )
    console.print(
        f"✅ Skeletal structure saved to [bold blue][link=file://{skeleton_file}]{skeleton_file.name}[/link][/bold blue]"
    )
    console.print(
        f"✅ Block tree saved to [bold blue][link=file://{tree_file}]{tree_file.name}[/link][/bold blue]"
    )


if __name__ == "__main__":
    main()
