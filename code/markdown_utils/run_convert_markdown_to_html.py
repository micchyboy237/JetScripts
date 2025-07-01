import os

from jet.code.markdown_utils import base_analyze_markdown
from jet.code.markdown_utils._converters import convert_markdown_to_html
from jet.code.markdown_utils._markdown_analyzer import get_summary
from jet.file.utils import save_file
from jet.utils.print_utils import print_dict_types

md_content1 = """
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

md_content2 = """
```python
def hello():
    print("Hello, World!")
```
| Header1 | Header2 |
|---------|---------|
| Cell1   | Cell2   |

The HTML specification is maintained by the W3C.

*[HTML]: HyperText Markup Language

A paragraph with custom attributes {#para1 .class1 style="color: blue;"}

Term 1
: Definition for term 1.

Term 2
: Definition for term 2.

```javascript
function greet() {
    console.log("Hello!");
}
```

Here is some text[^1].

[^1]: This is a footnote.

<div markdown="1">
*Emphasis* inside a div.
</div>

| Name | Age |
|------|-----|
| Alice | 25  |
| Bob   | 30  |

!!! warning "Custom Title"
    This is a warning admonition.
    It spans multiple lines.

```python
def calculate_sum(a, b):
    return a + b
```

A paragraph with legacy attributes {id="para1" class="class1"}.

This is *emphasized* and /italicized/ text.

title: My Document
author: John Doe
date: 2025-07-02

Content starts here.

1. Ordered item
   - Nested bullet
   - Another bullet
2. Another ordered item

# Heading 1
## Heading 2
### Heading 3

Go to [[Target Page]] and [[Another Page|Display Text]].

Some content

Basic **bold** text

```python
def example():
    pass
```
| Col1 | Col2 |
|------|------|
| A    | B    |

Text with footnote[^1].

[^1]: Footnote content.
"""

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(
        __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    converted_html_1 = convert_markdown_to_html(md_content1)
    converted_html_2 = convert_markdown_to_html(md_content2)

    save_file(converted_html_1, f"{output_dir}/converted_html_1.html")
    save_file(converted_html_2, f"{output_dir}/converted_html_2.html")
