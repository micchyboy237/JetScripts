from jet.logger import logger
from langchain_community.document_loaders.tsv import UnstructuredTSVLoader
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# TSV

>A [tab-separated values (TSV)](https://en.wikipedia.org/wiki/Tab-separated_values) file is a simple, text-based file format for storing tabular data.[3] Records are separated by newlines, and values within a record are separated by tab characters.

## `UnstructuredTSVLoader`

You can also load the table using the `UnstructuredTSVLoader`. One advantage of using `UnstructuredTSVLoader` is that if you use it in `"elements"` mode, an HTML representation of the table will be available in the metadata.
"""
logger.info("# TSV")


loader = UnstructuredTSVLoader(
    file_path="./example_data/mlb_teams_2012.csv", mode="elements"
)
docs = loader.load()

logger.debug(docs[0].metadata["text_as_html"])

logger.info("\n\n[DONE]", bright=True)