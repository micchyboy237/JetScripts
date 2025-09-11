from jet.logger import logger
from langchain_community.embeddings import InfinityEmbeddings
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
# Infinity

>[Infinity](https://github.com/michaelfeil/infinity) allows the creation of text embeddings.

## Text Embedding Model

There exists an infinity Embedding model, which you can access with
"""
logger.info("# Infinity")


"""
For a more detailed walkthrough of this, see [this notebook](/docs/integrations/text_embedding/infinity)
"""
logger.info("For a more detailed walkthrough of this, see [this notebook](/docs/integrations/text_embedding/infinity)")

logger.info("\n\n[DONE]", bright=True)