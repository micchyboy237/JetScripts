import os

from jet.code.markdown_utils import base_parse_markdown
from jet.file.utils import save_file
from jet.utils.print_utils import print_dict_types

md_content = """
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

##### Code Block
[code]
print("Hello, world!")
[/code]

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

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(
        __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    results_ignore_links = base_parse_markdown(md_content, ignore_links=True)
    results_with_links = base_parse_markdown(md_content, ignore_links=False)

    print_dict_types(results_with_links)

    save_file(results_with_links, f"{output_dir}/results_with_links.json")
    save_file(results_ignore_links, f"{output_dir}/results_ignore_links.json")
