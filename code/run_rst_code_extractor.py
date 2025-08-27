import os
from jet.code.rst_code_extractor import rst_to_code_blocks
from jet.file.utils import save_file
from jet.logger import logger

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

if __name__ == '__main__':
    rst_file = "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/TextBlob/docs/advanced_usage.rst"
    python_file_path = f'{OUTPUT_DIR}/code_output.py'
    code_blocks = rst_to_code_blocks(rst_file)

    # Write the extracted Python code to the specified .py file
    os.makedirs(os.path.dirname(python_file_path), exist_ok=True)
    python_code_blocks = [
        code_block for code_block in code_blocks if code_block['type'] == 'python']
    with open(python_file_path, 'w') as python_file:
        for idx, code_block in enumerate(python_code_blocks):
            python_file.write(
                f'# Code block {idx + 1}\n' + code_block['code'] + '\n\n')
    logger.success(f"Saved ({len(python_code_blocks)}) code blocks to {
        python_file_path}")
    save_file(python_code_blocks, f"{OUTPUT_DIR}/python_code_blocks.json")
