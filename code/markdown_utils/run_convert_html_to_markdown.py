import os
import shutil

from jet.code.markdown_utils._converters import convert_html_to_markdown
from jet.file.utils import save_file

html1 = """
<ul>
  <li>Item 1</li>
  <li>Item 2
    <ol>
      <li>Step 2.1</li>
      <li>Step 2.2
        <ul>
          <li>Sub-step 2.2.1</li>
          <li>Sub-step 2.2.2</li>
        </ul>
      </li>
    </ol>
  </li>
  <li>Item 3</li>
</ul>
"""

html2 = """
<ul>
  <li>Term 1
    <dl>
      <dt>Definition term</dt>
      <dd>Definition description</dd>
    </dl>
  </li>
  <li>Term 2
    <dl>
      <dt>Another term</dt>
      <dd>Another description</dd>
    </dl>
  </li>
</ul>
"""

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(
        __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)

    md_content1 = convert_html_to_markdown(html1, ignore_links=True)
    save_file(md_content1, f"{output_dir}/md_content1.md")

    md_content2 = convert_html_to_markdown(html2, ignore_links=True)
    save_file(md_content2, f"{output_dir}/md_content2.md")
