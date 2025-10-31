import os
import shutil

from jet.code.markdown_utils._converters import convert_html_to_markdown
from jet.file.utils import load_file, save_file
from jet.logger import logger

html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/playwright/generated/run_playwright_extract/top_isekai_anime_2025/https_gamerant_com_new_isekai_anime_2025/page.html"
html = load_file(html_file)

# html = """
# <dl>
#   <dt>Release Date</dt>
#   <dd>January 1, 2025</dd>
#   <dt>Platforms</dt>
#   <dd>PC</dd>
#   <dd>Mac</dd>
# </dl>
# """

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(
        __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(output_dir, ignore_errors=True)

    md_content = convert_html_to_markdown(html, ignore_links=True)
    # md_content = convert_dl_blocks_to_md(html)
    logger.gray("RESULT:")
    logger.success(md_content)
    save_file(md_content, f"{output_dir}/md_content.md")
