from jet.logger import CustomLogger
from llama_index.readers.file import HTMLTagReader
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/data_connectors/html_tag_reader.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# HTML Tag Reader

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# HTML Tag Reader")

# %pip install llama-index-readers-file

# !pip install llama-index

"""
### Download HTML file
"""
logger.info("### Download HTML file")

# %%bash
wget -e robots=off --no-clobber --page-requisites \
  --html-extension --convert-links --restrict-file-names=windows \
  --domains docs.ray.io --no-parent --accept=html \
  -P data/ https://docs.ray.io/en/master/ray-overview/installation.html


reader = HTMLTagReader(tag="section", ignore_no_id=True)
docs = reader.load_data(
    "data/docs.ray.io/en/master/ray-overview/installation.html"
)

for doc in docs:
    logger.debug(doc.metadata)

logger.info("\n\n[DONE]", bright=True)