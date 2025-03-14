import os
import glob
import shutil

from jet.logger import logger


# List of NLTK data directories
nltk_paths = [
    "/Users/jethroestrada/nltk_data",
    # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/.venv/nltk_data",
    # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/.venv/share/nltk_data",
    # "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/.venv/lib/nltk_data",
    "/usr/share/nltk_data",
    "/usr/local/share/nltk_data",
    "/usr/lib/nltk_data",
    "/usr/local/lib/nltk_data"
]

# Find and delete files or folders with "maxent_ne_chunker" in the name
found_paths = []

for path in nltk_paths:
    search_pattern = os.path.join(
        path, "chunkers", "maxent_ne_chunker*")  # Match any file or folder
    matches = glob.glob(search_pattern)

    if matches:
        for match in matches:
            if os.path.isfile(match):
                logger.success(f"Deleting file: {match}")
                os.remove(match)
            elif os.path.isdir(match):
                logger.success(f"Deleting folder: {match}")
                shutil.rmtree(match)
            found_paths.append(match)

if not found_paths:
    logger.error("No matching files or folders found.")

logger.info("Deletion process completed.")
