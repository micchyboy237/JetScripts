import argparse
import shutil
from pathlib import Path

from jet.code.markdown_utils._converters import base_convert_html_to_markdown
from jet.file.utils import load_file, save_file
from jet.logger import logger

DEFAULT_HTML_FILE = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/node_extraction/sample.html"

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main(html_file):
    html = load_file(html_file)

    # Run with ignore_links=True
    md_content_ignore_links = base_convert_html_to_markdown(html, ignore_links=True)
    logger.gray("RESULT (ignore_links=True):")
    logger.success(md_content_ignore_links)
    save_file(md_content_ignore_links, f"{OUTPUT_DIR}/md_content_ignore_links.md")

    # Run with ignore_links=False
    md_content_with_links = base_convert_html_to_markdown(html, ignore_links=False)
    logger.gray("RESULT (ignore_links=False):")
    logger.success(md_content_with_links)
    save_file(md_content_with_links, f"{OUTPUT_DIR}/md_content_with_links.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert an HTML file to Markdown.")
    parser.add_argument(
        "html_file",
        nargs="?",
        default=DEFAULT_HTML_FILE,
        help=f"Path to HTML file to convert (default: {DEFAULT_HTML_FILE})",
    )
    args = parser.parse_args()
    main(args.html_file)
