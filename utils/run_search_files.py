from jet.utils import search_files
from jet.logger import logger
import os
import shutil

if __name__ == '__main__':
    input_base_dirs = [
        "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/llama_index/docs/docs/understanding/putting_it_all_together",
    ]
    include_files = []
    exclude_files = []
    extensions = [".ipynb", ".md", ".mdx", ".rst"]
    output_dir = "generated"

    # Reset folder if exists
    shutil.rmtree(output_dir, ignore_errors=True)

    # Get files
    files = search_files(input_base_dirs, extensions,
                         include_files, exclude_files)
    logger.info(f"Found {len(files)} files with extensions {extensions}")

    # Process each file
    for base_dir in input_base_dirs:
        # Create output directory based on the last folder name in the base_dir
        base_name = os.path.basename(base_dir)
        output_base = os.path.join(output_dir, base_name)

        # Create the output base directory if it doesn't exist
        os.makedirs(output_base, exist_ok=True)

        # Process files for this base directory
        for file_path in files:
            if file_path.startswith(base_dir):
                # Get the relative path from the base directory
                rel_path = os.path.relpath(file_path, base_dir)
                # Construct the output path
                output_path = os.path.join(output_base, rel_path)
                # Create necessary subdirectories
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                # Copy the file
                shutil.copy2(file_path, output_path)
                logger.info(f"Copied {file_path} to {output_path}")
