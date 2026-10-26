import argparse
import os
import shutil

from jet.file.utils import load_file, save_file
from jet.scrapers.utils import extract_by_heading_hierarchy

DEFAULT_URL_OR_FILE = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/node_extraction/sample.html"
DEFAULT_EXCLUDES = ["head", "nav", "footer", "script", "style", "button"]


def main():
    parser = argparse.ArgumentParser(
        description="Extracts heading hierarchies, contents, and metadatas from an HTML file or URL."
    )
    parser.add_argument(
        "url_or_file",
        nargs="?",
        default=DEFAULT_URL_OR_FILE,
        help="Path to the HTML file or URL to process",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=os.path.join(
            os.path.dirname(__file__),
            "generated",
            os.path.splitext(os.path.basename(__file__))[0],
        ),
        help="Output directory for generated files",
    )
    parser.add_argument(
        "-e",
        "--excludes",
        nargs="*",
        default=DEFAULT_EXCLUDES,
        help="List of HTML tags to exclude during preprocessing (only applies to local files)",
    )
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    url_or_file = args.url_or_file
    excludes = args.excludes

    # Check if the input is a local file or a URL
    if os.path.isfile(url_or_file):
        # Load and preprocess local HTML file
        html_str: str = load_file(url_or_file)

        # from jet.code.html_utils import convert_dl_blocks_to_md, preprocess_html

        # html_str = convert_dl_blocks_to_md(html_str)
        # html_str = preprocess_html(html_str, excludes=excludes)

        save_file(html_str, f"{OUTPUT_DIR}/page.html")
    else:
        # For URLs, skip preprocessing and let extract_by_heading_hierarchy handle it
        # Save the URL as-is for reference
        save_file(url_or_file, f"{OUTPUT_DIR}/source_url.txt")

    # extract_by_heading_hierarchy accepts either a URL or an HTML file path
    headings = extract_by_heading_hierarchy(url_or_file)
    save_file(headings, f"{OUTPUT_DIR}/headings.json")

    contents = [heading.get_content() for heading in headings]
    save_file(contents, f"{OUTPUT_DIR}/contents.json")

    metadatas = [dict(node.get_element_details()) for node in headings]
    save_file(metadatas, f"{OUTPUT_DIR}/metadatas.json")


if __name__ == "__main__":
    main()
