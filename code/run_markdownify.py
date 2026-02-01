import os
import re
import shutil

from jet.file.utils import load_file, save_file
from markdownify import markdownify

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "generated",
    os.path.splitext(os.path.basename(__file__))[0],
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/libs/smolagents/tools/examples/generated/examples_visit_webpage_tool/visit_webpage_tool_logs/tool_visit_webpage/call_0003/page.html"
html = load_file(html_file)
md_content = markdownify(html).strip()
md_content = re.sub(r"\n{3,}", "\n\n", md_content)

save_file(md_content, f"{OUTPUT_DIR}/md_content.md")
