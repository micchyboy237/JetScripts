import os

from jet.code.markdown_utils import parse_markdown
from jet.code.markdown_utils._converters import convert_html_to_markdown
from jet.code.markdown_utils._markdown_parser import base_parse_markdown, derive_by_header_hierarchy
from jet.file.utils import load_file, save_file
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

Inline JSON Example of Metadata Scoring:

{
  "query": "quarterly revenue",
  "results": [
    {
      "doc_id": "doc_123",
      "content": "...",
      "metadata": { "source": "SEC Filing", "date": "2025-05-15" },
      "final_score": 0.95
    },
    {
      "doc_id": "doc_456",
      "content": "...",
      "metadata": { "source": "Internal Draft", "date": "2025-05-14" },
      "final_score": 0.78
    }
  ]
}
"""

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(
        __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    # md_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank_3/top_rag_strategies_reddit_2025/pages/medium_com_aa779_rag_in_2025_7_proven_strategies_to_deploy_retrieval_augmented_generation_at_scale_d1f71dfbfbba/page.md"
    # md_content = load_file(md_file)

    save_file(md_content, f"{output_dir}/md_content.md")

    markdown_tokens = base_parse_markdown(md_content)
    save_file(markdown_tokens, f"{output_dir}/markdown_tokens.json")

    results = derive_by_header_hierarchy(md_content)
    save_file(results, f"{output_dir}/results.json")
