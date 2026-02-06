BASE_DEFAULTS: BaseMarkdownAnalysis = {
    "analysis": Analysis(
        headers=0,
        paragraphs=0,
        blockquotes=0,
        code_blocks=0,
        ordered_lists=0,
        unordered_lists=0,
        tables=0,
        html_blocks=0,
        html_inline_count=0,
        words=0,
        characters=0,
        header_counts=HeaderCounts(h1=0, h2=0, h3=0, h4=0, h5=0, h6=0),
        text_links=0,
        image_links=0,
    ),
    "header": [],
    "paragraph": [],
    "blockquote": [],
    "code_block": [],
    "table": [],
    "unordered_list": [],
    "ordered_list": [],
    "text_link": [],
    "image_link": [],
    "footnotes": [],
    "inline_code": [],
    "emphasis": [],
    "task_items": [],
    "html_inline": [],
    "html_blocks": [],
    "tokens_sequential": [],  # Empty list is still valid with updated TokenSequential
}
import os
import tempfile
from pathlib import Path
from typing import Any, cast

from jet.code.html_utils import valid_html
from jet.code.markdown_types import MarkdownAnalysis, SummaryDict
from jet.code.markdown_types.base_markdown_analysis_types import (
    Analysis,
    BaseMarkdownAnalysis,
    CodeBlock,
    Emphasis,
    Footnote,
    Header,
    HeaderCounts,
    HtmlBlock,
    ImageLink,
    InlineCode,
    ListItem,
    Table,
    TaskItem,
    TextLink,
    TokenSequential,
)