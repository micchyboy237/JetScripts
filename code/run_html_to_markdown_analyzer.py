import os
from pathlib import Path
import shutil

from jet.code.html_to_markdown_analyzer import analyze_markdown_file, convert_html_to_markdown, process_html_for_analysis
from jet.file.utils import save_file

sample_html = """
<h1>Welcome</h1>
<p>This is a <a href="https://example.com">test</a>.</p>
<h2>Code Example</h2>
<pre><code>print("Hello")</code></pre>
"""

if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)

    # Example usage
    output_md_path = f"{output_dir}/markdown.md"
    output_md_path2 = f"{output_dir}/markdown2.md"
    output_analysis_path = f"{output_dir}/analysis.json"
    output_analysis_path2 = f"{output_dir}/analysis2.json"

    markdown_text = convert_html_to_markdown(sample_html)
    save_file(markdown_text, output_md_path)

    analysis = analyze_markdown_file(output_md_path)
    save_file(analysis, output_analysis_path)

    analysis2 = process_html_for_analysis(sample_html, output_md_path2)
    save_file(analysis2, output_analysis_path2)
