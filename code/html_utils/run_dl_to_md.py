import os
import shutil

from jet.code.html_utils import dl_to_md
from jet.file.utils import save_file
from jet.logger import logger

html = """
<dl><dt>A</dt><dd>1</dd></dl>
<p>---</p>
<dl><dt>B</dt><dd>2</dd><dd>3</dd></dl>
"""

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(
        __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)

    md_content = dl_to_md(html)
    logger.gray("RESULT:")
    logger.success(md_content)
    save_file(md_content, f"{output_dir}/md_content.md")
