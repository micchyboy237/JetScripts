from jet.code.markdown_utils import analyze_markdown, parse_markdown
from jet.code.splitter_markdown_utils import get_md_header_contents
from jet.file.utils import load_file
from jet.logger import logger


from typing import List, Dict

from jet.transformers.formatters import format_json


if __name__ == "__main__":
    md_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/__sample.md"

    results = parse_markdown(md_file)
    logger.gray("parse_markdown:")
    logger.success(results)

    results = analyze_markdown(md_file)
    logger.gray("analyze_markdown:")
    logger.success(results)

    md_content = load_file(md_file)
    results = get_md_header_contents(md_content, ignore_links=True)
    logger.gray("get_md_header_contents:")
    logger.success(format_json(results))
