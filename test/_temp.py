from typing import List, Literal, TypedDict
import markdown
import markdown.extensions as md_ext

# Define supported markdown extensions as Literal for type safety
MarkdownExtension = Literal[
    "extra",
    "abbr",
    "attr_list",
    "def_list",
    "fenced_code",
    "footnotes",
    "md_in_html",
    "tables",
    "admonition",
    "codehilite",
    "legacy_attrs",
    "legacy_em",
    "meta",
    "nl2br",
    "sane_lists",
    "smarty",
    "toc",
    "wikilinks",
]


class MarkdownExtensions(TypedDict):
    extensions: List[MarkdownExtension]


def render_markdown(md_content: str, exts: MarkdownExtensions) -> str:
    """
    Render markdown with supported extensions enabled.

    Args:
        md_content (str): Markdown content to render.
        exts (MarkdownExtensions): Dictionary containing a list of extension names.

    Returns:
        str: Rendered HTML output.
    """
    # Map extension names to their corresponding markdown extension paths
    extension_map = {
        "extra": "markdown.extensions.extra",
        "abbr": "markdown.extensions.abbr",
        "attr_list": "markdown.extensions.attr_list",
        "def_list": "markdown.extensions.def_list",
        "fenced_code": "markdown.extensions.fenced_code",
        "footnotes": "markdown.extensions.footnotes",
        "md_in_html": "markdown.extensions.md_in_html",
        "tables": "markdown.extensions.tables",
        "admonition": "markdown.extensions.admonition",
        "codehilite": "markdown.extensions.codehilite",
        "legacy_attrs": "markdown.extensions.legacy_attrs",
        "legacy_em": "markdown.extensions.legacy_em",
        "meta": "markdown.extensions.meta",
        "nl2br": "markdown.extensions.nl2br",
        "sane_lists": "markdown.extensions.sane_lists",
        "smarty": "markdown.extensions.smarty",
        "toc": "markdown.extensions.toc",
        "wikilinks": "markdown.extensions.wikilinks",
    }

    # Validate and collect extensions
    valid_extensions = []
    for ext in exts.get("extensions", []):
        if ext in extension_map:
            valid_extensions.append(extension_map[ext])
        else:
            raise ValueError(f"Unsupported markdown extension: {ext}")

    # Render markdown with specified extensions
    return markdown.markdown(md_content, extensions=valid_extensions)
