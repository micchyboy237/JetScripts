import os
import tempfile

from jet.cache.chromadb.data_utils import check_chromadb_data
from jet.logger import CustomLogger

# Set up logging
OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
log_file = os.path.join(OUTPUT_DIR, "chroma_verify.log")
logger = CustomLogger(log_file, overwrite=True)

if __name__ == "__main__":
    # Verify the 'preferences' collection (from the manual add example)
    check_chromadb_data(
        persistence_path=tempfile.gettempdir(), collection_name="preferences")
    # Verify the 'autogen_docs' collection (from the RAG example)
    check_chromadb_data()
