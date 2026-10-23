import argparse
import shutil
from pathlib import Path

from jet.file.utils import load_file, save_file
from jet.scrapers.utils import (
    get_flattened_parents_with_most_children,
)

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Format, preprocess, and analyze an HTML file."
    )
    parser.add_argument(
        "html_file",
        nargs="?",
        default="/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/node_extraction/sample.html",
        help="Path to the HTML file to process",
    )
    args = parser.parse_args()
    html_file = args.html_file

    html = load_file(html_file)

    flattened_parents_with_most_children = get_flattened_parents_with_most_children(
        html
    )
    save_file(
        flattened_parents_with_most_children,
        OUTPUT_DIR / "flattened_parents_with_most_children.json",
    )
