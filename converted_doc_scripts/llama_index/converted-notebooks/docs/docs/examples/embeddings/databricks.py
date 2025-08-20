from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.embeddings.databricks import DatabricksEmbedding
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Databricks Embeddings
If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Databricks Embeddings")

# %pip install llama-index
# %pip install llama-index-embeddings-databricks


os.environ["DATABRICKS_TOKEN"] = "<MY TOKEN>"
os.environ["DATABRICKS_SERVING_ENDPOINT"] = "<MY ENDPOINT>"
embed_model = DatabricksEmbedding(model="databricks-bge-large-en")
Settings.embed_model = embed_model

embeddings = embed_model.get_text_embedding(
    "The DatabricksEmbedding integration works great."
)

logger.info("\n\n[DONE]", bright=True)