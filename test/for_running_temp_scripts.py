import os
from jet.logger import logger
from llama_index.core import SimpleDirectoryReader


INPUT_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
# COMBINED_FILE_NAME = "combined.txt"

# Combine file contents
documents = SimpleDirectoryReader(INPUT_DIR, required_exts=[".md"]).load_data()
texts = [doc.text for doc in documents]
combined_texts_str = "\n\n\n".join(texts)

logger.success(combined_texts_str)

# combined_file_path = os.path.join(INPUT_DIR, COMBINED_FILE_NAME)
# with open(combined_file_path, "w") as f:
#     f.write(combined_texts_str)
