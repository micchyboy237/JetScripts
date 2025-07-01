import pytest
from typing import List
# Adjust import based on your module structure
from _temp import render_markdown, MarkdownExtensions


class TestMarkdownRenderer:
    # Existing tests (from your input) are assumed to be included here.
    # I'll continue from test_admonition_extension and add the remaining tests.

    def test_admonition_extension(self):
        # Given: Markdown with admonition including edge case with multiple lines and type
        md_content = """
!!! warning "Custom Title"
    This is a warning admonition.
    It spans multiple lines.
"""
        exts: MarkdownExtensions = {"extensions": ["admonition"]}
        expected = (
            '<div class="admonition warning">\n'
            '<p class="admonition-title">Custom Title</p>\n'
            '<p>This is a warning admonition.\n'
            'It spans multiple lines.</p>\n'
            '</div>'
        )

        # When: Rendering markdown with admonition extension
        result = render_markdown(md_content, exts)

        # Then: The output includes admonition markup with custom title
        assert result.strip() == expected.strip()

    def test_codehilite_extension(self):
        # Given: Markdown with code block and language for syntax highlighting
        md_content = """
```python
def calculate_sum(a, b):
    return a + b
```
"""
        exts: MarkdownExtensions = {"extensions": ["codehilite"]}
        expected = (
            '<div class="codehilite"><pre><span></span><code><span class="k">def</span> '
            '<span class="nf">calculate_sum</span><span class="p">(</span><span class="n">a</span><span class="p">, </span>'
            '<span class="n">b</span><span class="p">):</span>\n'
            '    <span class="k">return</span> <span class="n">a</span> <span class="o">+</span> <span class="n">b</span>\n'
            '</code></pre></div>'
        )

        # When: Rendering markdown with codehilite extension
        result = render_markdown(md_content, exts)

        # Then: The output includes syntax-highlighted code block
        assert result.strip() == expected.strip()

    def test_legacy_attrs_extension(self):
        # Given: Markdown with legacy attribute syntax
        md_content = """
A paragraph with legacy attributes {id="para1" class="class1"}.
"""
        exts: MarkdownExtensions = {"extensions": ["legacy_attrs"]}
        expected = '<p id="para1" class="class1">A paragraph with legacy attributes.</p>'

        # When: Rendering markdown with legacy_attrs extension
        result = render_markdown(md_content, exts)

        # Then: The output includes attributes in legacy format
        assert result.strip() == expected.strip()

    def test_legacy_em_extension(self):
        # Given: Markdown with legacy emphasis syntax
        md_content = """
This is *emphasized* and _italicized_ text.
"""
        exts: MarkdownExtensions = {"extensions": ["legacy_em"]}
        expected = '<p>This is <em>emphasized</em> and <i>italicized</i> text.</p>'

        # When: Rendering markdown with legacy_em extension
        result = render_markdown(md_content, exts)

        # Then: The output uses <em> and <i> tags for emphasis
        assert result.strip() == expected.strip()

    def test_meta_extension(self):
        # Given: Markdown with metadata and edge case with multiple keys
        md_content = """
title: My Document
author: John Doe
date: 2025-07-02

Content starts here.
"""
        exts: MarkdownExtensions = {"extensions": ["meta"]}
        expected = '<p>Content starts here.</p>'

        # When: Rendering markdown with meta extension
        result = render_markdown(md_content, exts)

        # Then: The output strips metadata and renders content
        assert result.strip() == expected.strip()

    def test_nl2br_extension(self):
        # Given: Markdown with newlines that should convert to <br> tags
        md_content = """
Line one
Line two
Line three
"""
        exts: MarkdownExtensions = {"extensions": ["nl2br"]}
        expected = '<p>Line one<br />\nLine two<br />\nLine three</p>'

        # When: Rendering markdown with nl2br extension
        result = render_markdown(md_content, exts)

        # Then: The output includes <br> tags for newlines
        assert result.strip() == expected.strip()

    def test_sane_lists_extension(self):
        # Given: Markdown with mixed list types (edge case: nested and mixed lists)
        md_content = """
1. Ordered item
   - Nested bullet
   - Another bullet
2. Another ordered item
"""
        exts: MarkdownExtensions = {"extensions": ["sane_lists"]}
        expected = (
            '<ol>\n'
            '<li>Ordered item\n'
            '<ul>\n'
            '<li>Nested bullet</li>\n'
            '<li>Another bullet</li>\n'
            '</ul>\n'
            '</li>\n'
            '<li>Another ordered item</li>\n'
            '</ol>'
        )

        # When: Rendering markdown with sane_lists extension
        result = render_markdown(md_content, exts)

        # Then: The output handles mixed lists correctly
        assert result.strip() == expected.strip()

    def test_smarty_extension(self):
        # Given: Markdown with smart typography (e.g., quotes, dashes)
        md_content = """
"Quotes" and -- to en-dash, --- to em-dash.
"""
        exts: MarkdownExtensions = {"extensions": ["smarty"]}
        expected = '<p>&ldquo;Quotes&rdquo; and &ndash; to en-dash, &mdash; to em-dash.</p>'

        # When: Rendering markdown with smarty extension
        result = render_markdown(md_content, exts)

        # Then: The output includes smart typography entities
        assert result.strip() == expected.strip()

    def test_toc_extension(self):
        # Given: Markdown with headings for table of contents
        md_content = """
# Heading 1
## Heading 2
### Heading 3
"""
        exts: MarkdownExtensions = {"extensions": ["toc"]}
        expected = (
            '<div class="toc">\n'
            '<ul>\n'
            '<li><a href="#heading-1">Heading 1</a>\n'
            '<ul>\n'
            '<li><a href="#heading-2">Heading 2</a>\n'
            '<ul>\n'
            '<li><a href="#heading-3">Heading 3</a></li>\n'
            '</ul>\n'
            '</li>\n'
            '</ul>\n'
            '</li>\n'
            '</ul>\n'
            '</div>\n'
            '<h1 id="heading-1">Heading 1</h1>\n'
            '<h2 id="heading-2">Heading 2</h2>\n'
            '<h3 id="heading-3">Heading 3</h3>'
        )

        # When: Rendering markdown with toc extension
        result = render_markdown(md_content, exts)

        # Then: The output includes table of contents with correct hierarchy
        assert result.strip() == expected.strip()

    def test_wikilinks_extension(self):
        # Given: Markdown with wiki-style links
        md_content = """
Go to [[Target Page]] and [[Another Page|Display Text]].
"""
        exts: MarkdownExtensions = {"extensions": ["wikilinks"]}
        expected = (
            '<p>Go to <a class="wikilink" href="/Target-Page">Target Page</a> '
            'and <a class="wikilink" href="/Another-Page">Display Text</a>.</p>'
        )

        # When: Rendering markdown with wikilinks extension
        result = render_markdown(md_content, exts)

        # Then: The output includes wiki-style links
        assert result.strip() == expected.strip()

    def test_invalid_extension(self):
        # Given: Markdown content with an invalid extension
        md_content = "Some content"
        exts: MarkdownExtensions = {"extensions": ["invalid_ext"]}

        # When: Rendering markdown with an invalid extension
        # Then: It raises a ValueError
        with pytest.raises(ValueError, match="Unsupported markdown extension: invalid_ext"):
            render_markdown(md_content, exts)

    def test_empty_extensions(self):
        # Given: Markdown content with no extensions
        md_content = "Basic **bold** text"
        exts: MarkdownExtensions = {"extensions": []}
        expected = '<p>Basic <strong>bold</strong> text</p>'

        # When: Rendering markdown without extensions
        result = render_markdown(md_content, exts)

        # Then: The output renders basic markdown correctly
        assert result.strip() == expected.strip()

    def test_multiple_extensions(self):
        # Given: Markdown with multiple extension features (tables, fenced code, footnotes)
        md_content = """
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
        exts: MarkdownExtensions = {"extensions": [
            "fenced_code", "tables", "footnotes"]}
        expected = (
            '<pre><code class="language-python">def example():\n'
            '    pass\n'
            '</code></pre>\n'
            '<table>\n'
            '<thead>\n'
            '<tr>\n'
            '<th>Col1</th>\n'
            '<th>Col2</th>\n'
            '</tr>\n'
            '</thead>\n'
            '<tbody>\n'
            '<tr>\n'
            '<td>A</td>\n'
            '<td>B</td>\n'
            '</tr>\n'
            '</tbody>\n'
            '</table>\n'
            '<p>Text with footnote<sup id="fnref:1"><a class="footnote-ref" href="#fn:1">1</a></sup>.</p>\n'
            '<div class="footnote">\n'
            '<hr />\n'
            '<ol>\n'
            '<li id="fn:1">\n'
            '<p>Footnote content.<a class="footnote-backref" href="#fnref:1">↩</a></p>\n'
            '</li>\n'
            '</ol>\n'
            '</div>'
        )

        # When: Rendering markdown with multiple extensions
        result = render_markdown(md_content, exts)

        # Then: The output combines all extension features correctly
        assert result.strip() == expected.strip()
