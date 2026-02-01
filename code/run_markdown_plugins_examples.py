# demo_mdit_py_plugins.py
# Usage examples for mdit-py-plugins (latest as of 2026: v0.5.0)
# Requires: pip install markdown-it-py mdit-py-plugins

import shutil
from pathlib import Path

from markdown_it import MarkdownIt
from mdit_py_plugins.admon import admon_plugin
from mdit_py_plugins.amsmath import amsmath_plugin
from mdit_py_plugins.anchors import anchors_plugin
from mdit_py_plugins.attrs import attrs_block_plugin, attrs_plugin
from mdit_py_plugins.container import container_plugin
from mdit_py_plugins.deflist import deflist_plugin
from mdit_py_plugins.dollarmath import dollarmath_plugin
from mdit_py_plugins.field_list import fieldlist_plugin  # if you want RST-style fields
from mdit_py_plugins.footnote import footnote_plugin
from mdit_py_plugins.front_matter import front_matter_plugin
from mdit_py_plugins.myst_blocks import myst_block_plugin
from mdit_py_plugins.myst_role import myst_role_plugin
from mdit_py_plugins.subscript import sub_plugin
from mdit_py_plugins.tasklists import tasklists_plugin
from mdit_py_plugins.texmath import texmath_plugin
from mdit_py_plugins.wordcount import wordcount_plugin

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_and_save_demo(
    demo_number: int,
    demo_name: str,
    md: MarkdownIt,
    text: str,
    output_dir: Path = Path("demo_outputs"),
) -> None:
    """Run demo, print to console, and save structured output to file."""
    output_dir.mkdir(exist_ok=True, parents=True)

    # Clean filename: lowercase, replace spaces/_ with -, remove "demo_plugin_"
    safe_name = demo_name.replace("demo_plugin_", "").replace(" ", "_").lower()
    filename = f"{demo_number:02d}_{safe_name}.md"
    filepath = output_dir / filename

    tokens = md.parse(text)
    html = md.render(text)

    # Console output (same look as before)
    print(f"\n=== {demo_number:02d} | {demo_name} ===")
    print("Input Markdown:")
    print(text.strip())
    print("\nTokens:")
    for tok in tokens:
        print(f"  - {tok}")
    print("\nRendered HTML:")
    print(html.strip())
    print("=" * 70)

    # File content
    content = f"""# {demo_name}

## Input Markdown

```markdown
{text.rstrip()}
```

## Tokens

```
{"\n".join(f"- {tok}" for tok in tokens)}
```

## Rendered HTML

```html
{html.strip()}
```
"""

    filepath.write_text(content, encoding="utf-8")
    print(f"→ Saved to: {filepath}\n")


def demo_plugin_front_matter():
    md = MarkdownIt().use(front_matter_plugin)
    text = """---
title: Demo Document
author: Grok
date: 2026-02-02
---
# Hello
Some content.
"""
    return md, text


def demo_plugin_footnotes():
    md = MarkdownIt().use(footnote_plugin, inline=True)
    text = """Here is a reference[^1] and an inline footnote^[This is inline!].

[^1]: This is a regular footnote with **bold** support.
"""
    return md, text


def demo_plugin_deflist():
    md = MarkdownIt().use(deflist_plugin)
    text = """Apple
: A fruit that grows on trees.

Computer
: An electronic device.
~ Also can mean a person who computes (archaic).
"""
    return md, text


def demo_plugin_tasklists():
    md = MarkdownIt().use(tasklists_plugin, enabled=True, label=True)
    text = """- [ ] Buy milk
- [x] Write code
- [ ] Deploy to prod
"""
    return md, text


def demo_plugin_heading_anchors():
    md = MarkdownIt().use(anchors_plugin, permalink=True, permalinkSymbol="¶")
    text = """# Main Title

## Subsection One

Some text.
"""
    return md, text


def demo_plugin_dollar_math():
    md = MarkdownIt().use(dollarmath_plugin, allow_labels=True, double_inline=False)
    text = r"""Inline math: $E = mc^2$

Block math with label:
$$ \int_a^b f(x)\,dx = F(b) - F(a) \tag{1} $$
"""
    return md, text


def demo_plugin_admonitions():
    md = MarkdownIt().use(admon_plugin)
    text = """!!! note "Important"
    This is an admonition note.

??? tip "Collapsible Tip"
    Hidden details here.
"""
    return md, text


def demo_plugin_attrs():
    md = MarkdownIt().use(attrs_plugin).use(attrs_block_plugin)
    text = """{#intro .lead}
A paragraph with block attrs.

Image with inline attrs: ![alt](https://example.com/img.jpg){#myimg .rounded width=300}
"""
    return md, text


def demo_plugin_custom_container():
    md = MarkdownIt().use(
        container_plugin,
        name="spoiler",
        marker=":",
        validate=lambda marker, info: info.strip().split(None, 1)[0] == "spoiler",
        render=lambda tokens, idx, options, env: (
            f"<details><summary>Spoiler</summary>\n"
            f"{md.renderer.render(tokens[idx + 1].children or [], options, env)}\n"
            f"</details>"
        )
        if tokens[idx].nesting == 1
        else "</details>",
    )
    text = """::: spoiler Optional title here
This is **hidden** content with *formatting*!

- List item
- Another
:::
"""
    return md, text


def demo_plugin_wordcount():
    md = MarkdownIt()
    md.use(wordcount_plugin, per_minute=200)  # or wordcount_plugin(md, per_minute=200)

    text = """This is a short paragraph with about fifteen words.

Another one here with **bold** and *italic* text.
"""
    # Note: The plugin consumes and puts wordcount info into tokens' env
    return md, text


def demo_plugin_texmath():
    md = MarkdownIt().use(
        texmath_plugin, delimiters="dollars"
    )  # or "brackets", "gitlab", etc.
    text = r"""Inline: \( E = mc^2 \)

Display: \[ \sum_{i=1}^n i = \frac{n(n+1)}{2} \]

With equation number: \[ a^2 + b^2 = c^2 \tag{Pythagoras} \]
"""
    return md, text


def demo_plugin_amsmath():
    md = MarkdownIt().use(amsmath_plugin)
    text = r"""\begin{align}
    a_1 &= b_1 + c_1 \\
    a_2 &= b_2 + c_2 - d_2 + e_2
\end{align}

\begin{gather*}
  x &= 1 + 2 + 3 \\
  y &= 4 + 5
\end{gather*}
"""
    return md, text


def demo_plugin_myst_role():
    md = MarkdownIt().use(myst_role_plugin)
    text = """This is a {ref}`cross-reference`.

Inline {kbd}`Ctrl + C` to copy.

{abbr}`HTML (HyperText Markup Language)` is fun.
"""
    return md, text


def demo_plugin_myst_blocks():
    md = MarkdownIt().use(myst_block_plugin)
    text = """(my-target)=
# Section with target

% This is a MyST comment (not rendered)

+++

A block break above
"""
    return md, text


def demo_plugin_fieldlist():
    md = MarkdownIt().use(fieldlist_plugin)
    text = """:Date: 2026-02-02
:Author: Grok
:Version: 1.0

Some content after the fields.
"""
    return md, text


def demo_plugin_subscript():
    md = MarkdownIt().use(sub_plugin)
    text = """H~2~O is water.

x~i~ = 5
"""
    return md, text


if __name__ == "__main__":
    print("Running mdit-py-plugins demo examples...\n")

    demos = [
        ("plugin_front_matter", demo_plugin_front_matter),
        ("plugin_footnotes", demo_plugin_footnotes),
        ("plugin_deflist", demo_plugin_deflist),
        ("plugin_tasklists", demo_plugin_tasklists),
        ("plugin_heading_anchors", demo_plugin_heading_anchors),
        ("plugin_dollar_math", demo_plugin_dollar_math),
        ("plugin_admonitions", demo_plugin_admonitions),
        ("plugin_attrs", demo_plugin_attrs),
        ("plugin_custom_container", demo_plugin_custom_container),
        ("plugin_subscript", demo_plugin_subscript),
        ("plugin_wordcount", demo_plugin_wordcount),
        ("plugin_texmath", demo_plugin_texmath),
        ("plugin_amsmath", demo_plugin_amsmath),
        ("plugin_myst_role", demo_plugin_myst_role),
        ("plugin_myst_blocks", demo_plugin_myst_blocks),
        ("plugin_fieldlist", demo_plugin_fieldlist),
    ]

    for i, (short_name, demo_func) in enumerate(demos, start=1):
        print(f"→ Running demo {i:02d}/{len(demos)}: {short_name}")
        md, text = demo_func()
        full_name = f"demo_{short_name}"
        run_and_save_demo(i, full_name, md, text, OUTPUT_DIR)
