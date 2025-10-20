import os
import shutil

from jet.code.markdown_utils._converters import convert_html_to_markdown
from jet.file.utils import save_file

sample_html = """
<p>Visit <a href="https://example.com" title="Example Site Tooltip">this site</a> for more info.</p>
<ul class="table-content-list">
    <li class="table-content-element icon">
        <div class="table-content-link">
            <a href="#2025-isekai-anime-sequels">
                2025 Isekai Anime Sequels
            </a>
        </div>
    </li>
</ul>
"""

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(
        __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)

    converted_markdown = convert_html_to_markdown(sample_html)
    converted_markdown_no_links = convert_html_to_markdown(sample_html, ignore_links=True)

    save_file(converted_markdown, f"{output_dir}/converted_markdown.md")
    save_file(converted_markdown_no_links, f"{output_dir}/converted_markdown_no_links.md")
