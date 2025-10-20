import os

from jet.code.markdown_utils._preprocessors import is_markdown_link
from jet.file.utils import save_file
from jet.logger import logger

text_with_sample_md_links = """
Visit [Google](https://www.google.com) now
"""

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(
        __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    result = is_markdown_link(text_with_sample_md_links)

    logger.log("Result: ", result, colors=["GRAY", "SUCCESS"])
    save_file(result, f"{output_dir}/result.txt")
