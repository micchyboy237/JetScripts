from jet.code.markdown_utils import analyze_markdown, parse_markdown
from jet.code.splitter_markdown_utils import get_md_header_contents
from jet.file.utils import load_file
from jet.logger import logger


from typing import List, Dict

from jet.transformers.formatters import format_json

md_content = "###### [Game Rant logo](https://screenrant.com/db/)\n\nExpand"

if __name__ == "__main__":

    results = parse_markdown(md_content)
    logger.gray("parse_markdown:")
    logger.success(results)

    results = analyze_markdown(md_content)
    logger.gray("analyze_markdown:")
    logger.success(results)

    # md_content = load_file(md_content)

    results = get_md_header_contents(md_content, ignore_links=True)
    results = [{
        "header": result["header"],
        "content": result["content"],
        "text": result["text"],
    } for result in results]
    expected = [
        {
            "header": "###### Game Rant logo",
            "content": "\nExpand",
            "text": "###### Game Rant logo\n\nExpand"
        }
    ]
    logger.gray("results:")
    logger.success(format_json(results))
    logger.gray("expected:")
    logger.success(format_json(expected))
    assert results == expected
