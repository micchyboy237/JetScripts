from jet.utils.commands import copy_to_clipboard
import markdown
import html2text
import logging
import re

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Singleton HTML2Text instance for performance
_html2text_instance = None

def get_html2text_instance() -> html2text.HTML2Text:
    """Returns a configured HTML2Text singleton instance."""
    global _html2text_instance
    if _html2text_instance is None:
        _html2text_instance = html2text.HTML2Text()
        _html2text_instance.ignore_images = False  # Allow alt text processing
        _html2text_instance.ignore_links = True
        _html2text_instance.body_width = 0
        _html2text_instance.ignore_tables = False
        _html2text_instance.skip_internal_links = True
        _html2text_instance.ignore_emphasis = True
        _html2text_instance.inline_links = False
        _html2text_instance.use_automatic_links = False
        _html2text_instance.unicode_snob = True
        _html2text_instance.ignore_anchors = True
        _html2text_instance.default_image_alt = " "
        _html2text_instance.pad_tables = True
        _html2text_instance.single_line_break = True
    return _html2text_instance

def md_to_plain_text(md_content: str, ignore_images: bool = True, ignore_links: bool = True) -> str:
    """
    Converts Markdown to plain text.

    Args:
        md_content: Input Markdown string.
        ignore_images: If True, replace images with alt text or empty.
        ignore_links: If True, show link text only (not URLs).

    Returns:
        Plain text string.

    Raises:
        TypeError: If md_content is not a string.
        ValueError: If md_content is empty or None.
    """
    if not isinstance(md_content, str):
        logger.error("Input must be a string")
        raise TypeError("Input must be a string")
    if not md_content or md_content.isspace():
        logger.warning("Empty or whitespace-only Markdown input")
        raise ValueError("Markdown content cannot be empty or whitespace")

    # Step 1: MD → HTML
    extensions = [
        "markdown.extensions.extra",
        "markdown.extensions.abbr",
        "markdown.extensions.attr_list",
        "markdown.extensions.def_list",
        "markdown.extensions.fenced_code",
        "markdown.extensions.footnotes",
        "markdown.extensions.md_in_html",
        "markdown.extensions.tables",
        "markdown.extensions.admonition",
        "markdown.extensions.codehilite",
        "markdown.extensions.legacy_attrs",
        "markdown.extensions.legacy_em",
        "markdown.extensions.meta",
        "markdown.extensions.nl2br",
        "markdown.extensions.sane_lists",
        "markdown.extensions.smarty",
        "markdown.extensions.toc",
        "markdown.extensions.wikilinks",
    ]
    html = markdown.markdown(md_content, extensions=extensions)
    logger.debug(f"Intermediate HTML: {html}")

    # Step 2: HTML → Plain text
    h = get_html2text_instance()
    h.ignore_images = ignore_images
    h.ignore_links = ignore_links
    h.default_image_alt = "" if ignore_images else " "
    plain_text = h.handle(html)
    logger.debug(f"Raw plain text: {plain_text}")

    # Step 3: Post-process to clean up
    lines = plain_text.splitlines()
    cleaned_lines = []
    skip_footnote_section = False
    previous_line_is_block = False
    for line in lines:
        # Skip footnote definitions section
        if line.strip().startswith("* * *"):
            skip_footnote_section = True
            continue
        if skip_footnote_section and re.match(r'^\s*\d+\.\s+', line):
            continue
        # Skip image reference definitions
        if re.match(r'^\[\d+\]:\s+', line):
            continue
        # Clean headers
        line = re.sub(r'^#+\s+', '', line)
        # Clean image references
        line = re.sub(r'!\[([^\]]*)\]\[\d+\]', r'\1', line)
        # Clean footnote references
        line = re.sub(r'\[\^(\d+)\]', r'[\1]', line)
        # Remove footnote return arrows
        line = line.replace(' ↩', '')
        # Remove blockquote markers
        line = line.lstrip('> ')
        # Clean table pipes
        line = re.sub(r'^\|\s*', '', line)
        line = re.sub(r'\s*\|\s*$', '', line)
        line = re.sub(r'\s*\|\s*', '  ', line)
        # Fix escaped list markers
        line = line.replace('\\-', '*')
        # Remove table separator lines
        if re.match(r'^-+\s*-+\s*$', line.strip()):
            continue
        # Add blank line before new block (header, paragraph, list)
        if line.strip() and cleaned_lines and not previous_line_is_block and not line.startswith('  '):
            cleaned_lines.append("")
        # Split blockquote lists
        if line.strip().startswith('Note:') and '*' in line:
            parts = line.split(' * ')
            cleaned_lines.append(parts[0])
            for part in parts[1:]:
                if part.strip():
                    cleaned_lines.append(f"* {part}")
            previous_line_is_block = False
            continue
        cleaned_lines.append(line)
        previous_line_is_block = bool(line.strip() and not line.startswith('  '))

    # Join lines and reduce consecutive newlines to two
    plain_text = "\n".join(line.rstrip() for line in cleaned_lines if line.strip() or line == "")
    plain_text = re.sub(r'\n{3,}', '\n\n', plain_text)
    logger.debug(f"Cleaned plain text: {plain_text}")

    return plain_text.strip()

# Usage example
if __name__ == "__main__":
    md = """
Sample title

# Project Overview
Welcome to our **project**! This is an `introduction` to our work, featuring a [website](https://project.com).

![Project Logo](https://project.com/logo.png)

> **Note**: Always check the [docs](https://docs.project.com) for updates.

## Features
- [ ] Task 1: Implement login
- [x] Task 2: Add dashboard
- Task 3: Optimize performance

### Technical Details
```python
def greet(name: str) -> str:
    return f"Hello, {name}!"
```

#### API Endpoints
| Endpoint       | Method | Description           |
|----------------|--------|-----------------------|
| /api/users     | GET    | Fetch all users       |
| /api/users/{id}| POST   | Create a new user     |

##### Inline Code
Use `print("Hello")` for quick debugging.

###### Emphasis
*Italic*, **bold**, and ***bold italic*** text are supported.

<div class="alert">This is an HTML block.</div>
<span class="badge">New</span> inline HTML.

##### Footnote
Here's a simple footnote,[^1] and here's a longer one.[^bignote]

[^1]: This is the first footnote.

[^bignote]: Here's one with multiple paragraphs and code.

    Indent paragraphs to include them in the footnote.

    `{ my code }`

    Add as many paragraphs as you like.

## Unordered list
- List item 1
    - Nested item
- List item 2
- List item 3

## Ordered list
1. Ordered item 1
2. Ordered item 2
3. Ordered item 3

## Inline HTML
<span class="badge">New</span> inline HTML
"""
    result = md_to_plain_text(md)

    expected_result = """Sample title

    # Project Overview

    Welcome to our **project**! This is an `introduction` to our work, featuring a website.

    Note: Always check the docs for updates.

    ## Features

    * [ ] Task 1: Implement login
    * [x] Task 2: Add dashboard
    * Task 3: Optimize performance

    ## Technical Details

    ```
    def greet(name: str) -> str:
        return f"Hello, {name}!"
    ```

    ## API Endpoints

    Endpoint       Method  Description
    /api/users     GET     Fetch all users
    /api/users/{id} POST   Create a new user

    ## Inline Code

    Use `print("Hello")` for quick debugging.

    ## Emphasis

    *Italic*, **bold**, and ***bold italic*** text are supported.

    This is an HTML block.

    New inline HTML.

    ## Unordered list

    * List item 1

    * Nested item
    * List item 2
    * List item 3

    ## Ordered list

    1. Ordered item 1
    2. Ordered item 2
    3. Ordered item 3

    ## Inline HTML

    New inline HTML"""

    logger.info("EXPECTED:\n" + expected_result)
    logger.info("RESULT:\n" + result)
    logger.debug("MATCH: %s", result == expected_result)
    copy_to_clipboard(result)
