import os
from pathlib import Path
import shutil

from jet.code.html_to_markdown_analyzer import analyze_markdown_file, convert_html_to_markdown, process_html_for_analysis
from jet.file.utils import save_file

sample_html = """
<div class="project-description">
  <h1>mrkdwn_analysis</h1>
  <p>
    <code>mrkdwn_analysis</code> is a powerful Python library designed to analyze Markdown files. It provides extensive parsing capabilities to extract and categorize various elements within a Markdown document, including headers, sections, links, images, blockquotes, code blocks, lists, tables, tasks (todos), footnotes, and even embedded HTML. This makes it a versatile tool for data analysis, content generation, or building other tools that work with Markdown.
  </p>
  <h2>Features</h2>
  <ul>
    <li>
      <p>
        <strong>File Loading</strong>: Load any given Markdown file by providing its file path.
      </p>
    </li>
    <li>
      <p>
        <strong>Header Detection</strong>: Identify all headers (ATX <code>#</code> to <code>######</code>, and Setext <code>===</code> and <code>---</code>) in the document, giving you a quick overview of its structure.
      </p>
    </li>
    <li>
      <p>
        <strong>Section Identification (Setext)</strong>: Recognize sections defined by a block of text followed by <code>=</code> or <code>-</code> lines, helping you understand the document’s conceptual divisions.
      </p>
    </li>
    <li>
      <p>
        <strong>Paragraph Extraction</strong>: Distinguish regular text (paragraphs) from structured elements like headers, lists, or code blocks, making it easy to isolate the body content.
      </p>
    </li>
    <li>
      <p>
        <strong>Blockquote Identification</strong>: Extract all blockquotes defined by lines starting with <code>&gt;</code>.
      </p>
    </li>
    <li>
      <p>
        <strong>Code Block Extraction</strong>: Detect fenced code blocks delimited by triple backticks (```), optionally retrieve their language, and separate programming code from regular text.
      </p>
    </li>
    <li>
      <p>
        <strong>List Recognition</strong>: Identify both ordered and unordered lists, including task lists ( <code>- [ ]</code>, <code>- [x]</code>), and understand their structure and hierarchy.
      </p>
    </li>
    <li>
      <p>
        <strong>Tables (GFM)</strong>: Detect GitHub-Flavored Markdown tables, parse their headers and rows, and separate structured tabular data for further analysis.
      </p>
    </li>
    <li>
      <p>
        <strong>Links and Images</strong>: Identify text links ( <code>[text](url)</code>) and images ( <code>![alt](url)</code>), as well as reference-style links. This is useful for link validation or content analysis.
      </p>
    </li>
    <li>
      <p>
        <strong>Footnotes</strong>: Extract and handle Markdown footnotes ( <code>[^note1]</code>), providing a way to process reference notes in the document.
      </p>
    </li>
    <li>
      <p>
        <strong>HTML Blocks and Inline HTML</strong>: Handle HTML blocks ( <code>&lt;div&gt;...&lt;/div&gt;</code>) as a single element, and detect inline HTML elements ( <code>&lt;span style="..."&gt;... &lt;/span&gt;</code>) as a unified component.
      </p>
    </li>
    <li>
      <p>
        <strong>Front Matter</strong>: If present, extract YAML front matter at the start of the file.
      </p>
    </li>
    <li>
      <p>
        <strong>Counting Elements</strong>: Count how many occurrences of a certain element type (e.g., how many headers, code blocks, etc.).
      </p>
    </li>
    <li>
      <p>
        <strong>Textual Statistics</strong>: Count the number of words and characters (excluding whitespace). Get a global summary ( <code>analyse()</code>) of the document’s composition.
      </p>
    </li>
  </ul>
  <h2>Installation</h2>
  <p>Install <code>mrkdwn_analysis</code> from PyPI: </p>
  <pre lang="bash">pip
		<span class="w"></span>install
		<span class="w"></span>markdown-analysis

	</pre>
  <h2>Usage</h2>
  <p>Using <code>mrkdwn_analysis</code> is straightforward. Import <code>MarkdownAnalyzer</code>, create an instance with your Markdown file path, and then call the various methods to extract the elements you need. </p>
  <pre lang="python3">
		<span class="kn">from</span>
		<span class="w"></span>
		<span class="nn">mrkdwn_analysis</span>
		<span class="w"></span>
		<span class="kn">import</span>
		<span class="n">MarkdownAnalyzer</span>
		<span class="n">analyzer</span>
		<span class="o">=</span>
		<span class="n">MarkdownAnalyzer</span>
		<span class="p">(</span>
		<span class="s2">"path/to/document.md"</span>
		<span class="p">)</span>
		<span class="n">headers</span>
		<span class="o">=</span>
		<span class="n">analyzer</span>
		<span class="o">.</span>
		<span class="n">identify_headers</span>
		<span class="p">()</span>
		<span class="n">paragraphs</span>
		<span class="o">=</span>
		<span class="n">analyzer</span>
		<span class="o">.</span>
		<span class="n">identify_paragraphs</span>
		<span class="p">()</span>
		<span class="n">links</span>
		<span class="o">=</span>
		<span class="n">analyzer</span>
		<span class="o">.</span>
		<span class="n">identify_links</span>
		<span class="p">()</span>
		<span class="o">...</span>
	</pre>
  <h3>Example</h3>
  <p>Consider <code>example.md</code>: </p>
  <pre lang="markdown">---
title: "Python 3.11 Report"
author: "John Doe"

		<span class="gu">date: "2024-01-15"</span>
		<span class="gu">---</span>
		<span class="gh">Python 3.11</span>
		<span class="gh">===========</span>

A major 
		<span class="gs">**Python**</span> release with significant improvements...


		<span class="gu">### Performance Details</span>

```python
import math
print(math.factorial(10))

	</pre>
  <blockquote>
    <p>
      <em>Quote</em>: "Python 3.11 brings the speed we needed"
    </p>
  </blockquote>
  <div class="note">
    <p>HTML block example</p>
  </div>
  <p>This paragraph contains inline HTML: <span>Red text</span>. </p>
  <ul>
    <li>Unordered list: <ul>
        <li>A basic point</li>
        <li>
          <input type="checkbox" disabled=""> A task to do
        </li>
        <li>
          <input type="checkbox" checked="" disabled=""> A completed task
        </li>
      </ul>
    </li>
  </ul>
  <ol>
    <li>Ordered list item 1</li>
    <li>Ordered list item 2</li>
  </ol>
  <pre>
				<code>
After analysis:

```python
analyzer = MarkdownAnalyzer("example.md")

print(analyzer.identify_headers())
# {"Header": [{"line": X, "level": 1, "text": "Python 3.11"}, {"line": Y, "level": 3, "text": "Performance Details"}]}

print(analyzer.identify_paragraphs())
# {"Paragraph": ["A major **Python** release ...", "This paragraph contains inline HTML: ..."]}

print(analyzer.identify_html_blocks())
# [{"line": Z, "content": "&lt;div class=\"note\"&gt;\n  &lt;p&gt;HTML block example&lt;/p&gt;\n&lt;/div&gt;"}]

print(analyzer.identify_html_inline())
# [{"line": W, "html": "&lt;span style=\"color:red;\"&gt;Red text&lt;/span&gt;"}]

print(analyzer.identify_lists())
# {
#   "Ordered list": [["Ordered list item 1", "Ordered list item 2"]],
#   "Unordered list": [["A basic point", "A task to do [Task]", "A completed task [Task done]"]]
# }

print(analyzer.identify_code_blocks())
# {"Code block": [{"start_line": X, "content": "import math\nprint(math.factorial(10))", "language": "python"}]}

print(analyzer.analyse())
# {
#   'headers': 2,
#   'paragraphs': 2,
#   'blockquotes': 1,
#   'code_blocks': 1,
#   'ordered_lists': 2,
#   'unordered_lists': 3,
#   'tables': 0,
#   'html_blocks': 1,
#   'html_inline_count': 1,
#   'words': 42,
#   'characters': 250
# }
</code>
			</pre>
  <h3>Key Methods</h3>
  <ul>
    <li>
      <code>__init__(self, input_file)</code>: Load the Markdown from path or file object.
    </li>
    <li>
      <code>identify_headers()</code>: Returns all headers.
    </li>
    <li>
      <code>identify_sections()</code>: Returns setext sections.
    </li>
    <li>
      <code>identify_paragraphs()</code>: Returns paragraphs.
    </li>
    <li>
      <code>identify_blockquotes()</code>: Returns blockquotes.
    </li>
    <li>
      <code>identify_code_blocks()</code>: Returns code blocks with content and language.
    </li>
    <li>
      <code>identify_lists()</code>: Returns both ordered and unordered lists (including tasks).
    </li>
    <li>
      <code>identify_tables()</code>: Returns any GFM tables.
    </li>
    <li>
      <code>identify_links()</code>: Returns text and image links.
    </li>
    <li>
      <code>identify_footnotes()</code>: Returns footnotes used in the document.
    </li>
    <li>
      <code>identify_html_blocks()</code>: Returns HTML blocks as single tokens.
    </li>
    <li>
      <code>identify_html_inline()</code>: Returns inline HTML elements.
    </li>
    <li>
      <code>identify_todos()</code>: Returns task items.
    </li>
    <li>
      <code>count_elements(element_type)</code>: Counts occurrences of a specific element type.
    </li>
    <li>
      <code>count_words()</code>: Counts words in the entire document.
    </li>
    <li>
      <code>count_characters()</code>: Counts non-whitespace characters.
    </li>
    <li>
      <code>analyse()</code>: Provides a global summary (headers count, paragraphs count, etc.).
    </li>
  </ul>
  <h3>Checking and Validating Links</h3>
  <ul>
    <li>
      <code>check_links()</code>: Validates text links to see if they are broken (e.g., non-200 status) and returns a list of broken links.
    </li>
  </ul>
  <h3>Global Analysis Example</h3>
  <pre lang="python3">
				<span class="n">analysis</span>
				<span class="o">=</span>
				<span class="n">analyzer</span>
				<span class="o">.</span>
				<span class="n">analyse</span>
				<span class="p">()</span>
				<span class="nb">print</span>
				<span class="p">(</span>
				<span class="n">analysis</span>
				<span class="p">)</span>
				<span class="c1"># {</span>
				<span class="c1">#   'headers': X,</span>
				<span class="c1">#   'paragraphs': Y,</span>
				<span class="c1">#   'blockquotes': Z,</span>
				<span class="c1">#   'code_blocks': A,</span>
				<span class="c1">#   'ordered_lists': B,</span>
				<span class="c1">#   'unordered_lists': C,</span>
				<span class="c1">#   'tables': D,</span>
				<span class="c1">#   'html_blocks': E,</span>
				<span class="c1">#   'html_inline_count': F,</span>
				<span class="c1">#   'words': G,</span>
				<span class="c1">#   'characters': H</span>
				<span class="c1"># }</span>
			</pre>
  <h2>Contributing</h2>
  <p>Contributions are welcome! Feel free to open an issue or submit a pull request for bug reports, feature requests, or code improvements. Your input helps make <code>mrkdwn_analysis</code> more robust and versatile. </p>
</div>
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
