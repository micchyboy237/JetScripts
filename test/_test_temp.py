import pytest
from _temp_test import md_to_plain_text

class TestMdToPlainText:
    @pytest.fixture
    def setup_html2text(self):
        """Fixture to reset HTML2Text singleton between tests."""
        global _html2text_instance
        _html2text_instance = None
        yield
        _html2text_instance = None

    def test_basic_formatting(self, setup_html2text):
        # Given: Markdown with headers, bold, and links
        md = "# Title\nThis is **bold** [link](https://example.com)"
        expected = "Title\n\nThis is bold link"

        # When: Convert to plain text
        result = md_to_plain_text(md)

        # Then: Matches expected output
        assert result == expected

    def test_images_handling(self, setup_html2text):
        # Given: Markdown with image
        md = "![alt text](img.png)"
        expected = "alt text"

        # When: Convert with ignore_images=True
        result = md_to_plain_text(md, ignore_images=True)

        # Then: Image replaced with alt text
        assert result == expected

    def test_footnotes_handling(self, setup_html2text):
        # Given: Markdown with footnotes
        md = "Text with [^1] reference.\n[^1]: Footnote content"
        expected = "Text with [1] reference."

        # When: Convert to plain text
        result = md_to_plain_text(md)

        # Then: Footnote definition removed, reference inlined
        assert result == expected

    def test_blockquote_handling(self, setup_html2text):
        # Given: Markdown with blockquote
        md = "> This is a **blockquote**"
        expected = "This is a blockquote"

        # When: Convert to plain text
        result = md_to_plain_text(md)

        # Then: Blockquote marker removed
        assert result == expected

    def test_table_handling(self, setup_html2text):
        # Given: Markdown with table
        md = "| Name | Age |\n|------|-----|\n| Alice | 30 |"
        expected = "Name   Age\nAlice  30"

        # When: Convert to plain text
        result = md_to_plain_text(md)

        # Then: Table formatted cleanly
        assert result == expected

    def test_inline_html_handling(self, setup_html2text):
        # Given: Markdown with inline HTML
        md = "<span class=\"badge\">New</span> feature"
        expected = "New feature"

        # When: Convert to plain text
        result = md_to_plain_text(md)

        # Then: HTML tags stripped
        assert result == expected

    def test_invalid_input_type(self, setup_html2text):
        # Given: Invalid non-string input
        md = 123

        # When/Then: Raises TypeError
        with pytest.raises(TypeError, match="Input must be a string"):
            md_to_plain_text(md)

    def test_empty_input(self, setup_html2text):
        # Given: Empty string input
        md = ""

        # When/Then: Raises ValueError
        with pytest.raises(ValueError, match="Markdown content cannot be empty or whitespace"):
            md_to_plain_text(md)

    def test_complex_markdown(self, setup_html2text):
        # Given: Complex Markdown from user sample
        md = """
# Project Overview
Welcome to **project**! [website](https://project.com)
![Logo](logo.png)
> **Note**: Check [docs](https://docs.project.com).
- Task 1
- Task 2
[^1]: Footnote
<div class="alert">Alert</div>
"""
        expected = """Project Overview

Welcome to project! website
Logo
Note: Check docs.
* Task 1
* Task 2"""

        # When: Convert to plain text
        result = md_to_plain_text(md)

        # Then: Matches expected output
        assert result == expected