import os
import shutil

from jet.file.utils import save_file
from jet.code.markdown_utils._converters import convert_markdown_to_html

md_content1 = """
* Item 1
* Item 2
  * Sub-item 2.1
  * Sub-item 2.2
* Item 3
* Item 4
  1. Sub-item 4.1
  2. Sub-item 4.2
"""

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(
        __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)

    html1 = convert_markdown_to_html(md_content1)

    save_file(html1, f"{output_dir}/html1.html")
