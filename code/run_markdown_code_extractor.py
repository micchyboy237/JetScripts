from jet.code.markdown_code_extractor import MarkdownCodeExtractor
from jet.file.utils import load_file, save_file
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

md_content = load_file("/Users/jethroestrada/Desktop/External_Projects/AI/examples/Context-Engineering/00_COURSE/00_mathematical_foundations/00_introduction.md")

if __name__ == "__main__":
    extractor = MarkdownCodeExtractor()
    results = extractor.extract_code_blocks(md_content)
    save_file(results, f"{OUTPUT_DIR}/results.json")
