import os
import shutil
import tempfile
from mrkdwn_analysis import MarkdownAnalyzer
from pathlib import Path

from jet.file.utils import save_file

md_content = """
# Header 1
Paragraph text with a [Link](https://example.com).
## Header 2
```python
print("Test")
```
"""

if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)

    # Use tempfile to create a temporary file in the system temp directory
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as temp_file:
        temp_file.write(md_content)
        temp_md_path = Path(temp_file.name)

    try:
        analyzer = MarkdownAnalyzer(str(temp_md_path))

        headers = analyzer.identify_headers()
        paragraphs = analyzer.identify_paragraphs()
        blockquotes = analyzer.identify_blockquotes()
        code_blocks = analyzer.identify_code_blocks()
        lists = analyzer.identify_lists()
        tables = analyzer.identify_tables()
        links = analyzer.identify_links()
        footnotes = analyzer.identify_footnotes()
        inline_code = analyzer.identify_inline_code()
        emphasis = analyzer.identify_emphasis()
        task_items = analyzer.identify_task_items()
        html_blocks = analyzer.identify_html_blocks()
        html_inline = analyzer.identify_html_inline()
        tokens_sequential = analyzer.get_tokens_sequential()
        word_count = analyzer.count_words()
        char_count = analyzer.count_characters()
        analysis = analyzer.analyse()

        save_file(headers, f"{output_dir}/headers.json")
        save_file(paragraphs, f"{output_dir}/paragraphs.json")
        save_file(blockquotes, f"{output_dir}/blockquotes.json")
        save_file(code_blocks, f"{output_dir}/code_blocks.json")
        save_file(lists, f"{output_dir}/lists.json")
        save_file(tables, f"{output_dir}/tables.json")
        save_file(links, f"{output_dir}/links.json")
        save_file(footnotes, f"{output_dir}/footnotes.json")
        save_file(inline_code, f"{output_dir}/inline_code.json")
        save_file(emphasis, f"{output_dir}/emphasis.json")
        save_file(task_items, f"{output_dir}/task_items.json")
        save_file(html_blocks, f"{output_dir}/html_blocks.json")
        save_file(html_inline, f"{output_dir}/html_inline.json")
        save_file(tokens_sequential, f"{output_dir}/tokens_sequential.json")
        save_file({"word_count": word_count}, f"{output_dir}/word_count.json")
        save_file({"char_count": char_count}, f"{output_dir}/char_count.json")
        save_file(analysis, f"{output_dir}/analysis.json")
    finally:
        # Safely remove the temporary file
        if temp_md_path.exists():
            try:
                temp_md_path.unlink()
            except PermissionError:
                print(
                    f"Warning: Could not delete temporary file {temp_md_path}")
