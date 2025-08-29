from jet.logger import CustomLogger
from llama_index.core import VectorStoreIndex
from llama_index.readers.deeplake import DeepLakeReader
import os
import random
import shutil
import textwrap


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/data_connectors/DeepLakeReader.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# DeepLake Reader

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# DeepLake Reader")

# %pip install llama-index-readers-deeplake

# !pip install llama-index

# import getpass



# os.environ["OPENAI_API_KEY"] = getpass.getpass("open ai api key: ")

reader = DeepLakeReader()
query_vector = [random.random() for _ in range(1536)]
documents = reader.load_data(
    query_vector=query_vector,
    dataset_path="hub://activeloop/paul_graham_essay",
    limit=5,
)

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What was a hard moment for the author?")
logger.debug(textwrap.fill(str(response), 100))

logger.info("\n\n[DONE]", bright=True)