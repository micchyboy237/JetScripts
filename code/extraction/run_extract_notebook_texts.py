import os
import shutil
from jet.code.extraction.extract_notebook_texts import run_notebook_extraction
from jet.logger import logger

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    input_path = "/Users/jethroestrada/Desktop/External_Projects/AI/examples/GenAI_Agents"
    output_dir = f"{OUTPUT_DIR}/{os.path.basename(input_path)}"

    logger.info("Extracting texts from notebooks...")
    run_notebook_extraction(input_path, f"{output_dir}/docs",
                            include_code=True, merge_consecutive_code=True, save_as="md")

    logger.info("Extracting blocks from notebooks...")
    run_notebook_extraction(input_path, f"{output_dir}/code_blocks",
                            include_code=True, merge_consecutive_code=True, save_as="blocks")

    # logger.info("Extracting documentation markdown...")
    # run_text_extraction(input_path, output_dir, save_as="md")
