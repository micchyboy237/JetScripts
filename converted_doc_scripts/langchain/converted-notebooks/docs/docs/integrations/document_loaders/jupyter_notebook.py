from jet.logger import logger
from langchain_community.document_loaders import NotebookLoader
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
# Jupyter Notebook

>[Jupyter Notebook](https://en.wikipedia.org/wiki/Project_Jupyter#Applications) (formerly `IPython Notebook`) is a web-based interactive computational environment for creating notebook documents.

This notebook covers how to load data from a `Jupyter notebook (.ipynb)` into a format suitable by LangChain.
"""
logger.info("# Jupyter Notebook")


loader = NotebookLoader(
    "example_data/notebook.ipynb",
    include_outputs=True,
    max_output_length=20,
    remove_newline=True,
)

"""
`NotebookLoader.load()` loads the `.ipynb` notebook file into a `Document` object.

**Parameters**:

* `include_outputs` (bool): whether to include cell outputs in the resulting document (default is False).
* `max_output_length` (int): the maximum number of characters to include from each cell output (default is 10).
* `remove_newline` (bool): whether to remove newline characters from the cell sources and outputs (default is False).
* `traceback` (bool): whether to include full traceback (default is False).
"""

loader.load()

logger.info("\n\n[DONE]", bright=True)