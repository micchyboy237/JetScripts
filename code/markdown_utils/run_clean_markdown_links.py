import os

from jet.code.markdown_utils._preprocessors import clean_markdown_links
from jet.file.utils import save_file
from jet.logger import logger

text_with_sample_md_links = """
[ ](/)
[Database](/db/) [Threads](/threads/)
"""

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(
        __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])


    result = clean_markdown_links(text_with_sample_md_links)
    logger.success(f"Result: '{result}'")

    save_file(result, f"{output_dir}/result.txt")
