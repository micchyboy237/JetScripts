from jet.code.splitter_markdown_utils import get_header_text, get_md_header_contents
from jet.code.helpers.markdown_header_text_splitter import MarkdownHeaderTextSplitter
from jet.logger import logger
from jet.transformers.formatters import format_json

if __name__ == "__main__":
    md_text = """
# Level 1
## Level 2.1
Content
## Level 2.2
Content
### Level 3
Content
"""
    headers_to_split_on = [
        ("#", "h1"), ("##", "h2"), ("###", "h3"),
        ("####", "h4"), ("#####", "h5"), ("######", "h6"),
    ]
    headers_to_split_on += [(f"* {header}", tag)
                            for header, tag in headers_to_split_on]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on, strip_headers=False, return_each_line=False
    )
    md_header_splits = markdown_splitter.split_text(md_text)
    logger.newline()
    logger.gray("md_header_splits:")
    logger.success(format_json(md_header_splits))

    # headers = get_md_header_contents(md_text)
    # logger.newline()
    # logger.gray("headers:")
    # logger.success(format_json(headers))

    # header_text = get_header_text(md_text)
    # logger.newline()
    # logger.gray("header_text:")
    # logger.success(header_text)

    assert len(
        md_header_splits) == 2, "Should handle nested headers without content"
