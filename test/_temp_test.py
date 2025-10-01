import markdown
import html2text
import logging
from jet.logger import logger
from jet.utils.commands import copy_to_clipboard

# Singleton HTML2Text instance for performance
_html2text_instance = None

def get_html2text_instance() -> html2text.HTML2Text:
    """Returns a configured HTML2Text singleton instance."""
    global _html2text_instance
    if _html2text_instance is None:
        _html2text_instance = html2text.HTML2Text()
        _html2text_instance.ignore_images = True
        _html2text_instance.ignore_links = True
        _html2text_instance.body_width = 0
        _html2text_instance.ignore_tables = False
        _html2text_instance.skip_internal_links = True
        _html2text_instance.ignore_emphasis = False
        _html2text_instance.inline_links = False
        _html2text_instance.use_automatic_links = False
        _html2text_instance.unicode_snob = True
        _html2text_instance.ignore_anchors = True
        _html2text_instance.default_image_alt = ""
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
        logging.error("Input must be a string")
        raise TypeError("Input must be a string")
    if not md_content or md_content.isspace():
        logging.warning("Empty or whitespace-only Markdown input")
        raise ValueError("Markdown content cannot be empty or whitespace")

    # Step 1: MD → HTML
    html = markdown.markdown(md_content, extensions=['tables', 'fenced_code', 'footnotes'])

    # Step 2: HTML → Plain text
    h = get_html2text_instance()
    # Update instance settings if needed
    h.ignore_images = ignore_images
    h.ignore_links = ignore_links
    plain_text = h.handle(html)

    # Post-process to clean up footnotes and blockquotes
    lines = plain_text.splitlines()
    cleaned_lines = []
    skip_footnote = False
    for line in lines:
        # Skip footnote definitions
        if line.strip().startswith("[^"):
            skip_footnote = True
            continue
        # Replace footnote references with inline text
        if "[^" in line:
            line = line.replace("[^1]:", "[1]").replace("[^1]", "[1]")
        # Remove blockquote markers
        if line.startswith("> "):
            line = line[2:]
        cleaned_lines.append(line)
    plain_text = "\n".join(cleaned_lines).strip()

    return plain_text

# Usage
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

[^1]: This is a footnote reference.
[^1]: Footnote definition here.

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
logger.success("RESULT:\n" + result)
logger.debug("MATCH: %s", result == expected_result)
copy_to_clipboard(result)
