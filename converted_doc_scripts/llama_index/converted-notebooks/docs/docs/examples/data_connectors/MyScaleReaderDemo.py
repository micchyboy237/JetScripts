from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.myscale import MyScaleReader
import clickhouse_connect
import logging
import os
import random
import shutil
import sys


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/data_connectors/MyScaleReaderDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# MyScale Reader

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# MyScale Reader")

# %pip install llama-index-readers-myscale

# !pip install llama-index


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


host = "YOUR_CLUSTER_HOST"
username = "YOUR_USERNAME"
password = "YOUR_CLUSTER_PASSWORD"
client = clickhouse_connect.get_client(
    host=host, port=8443, username=username, password=password
)


reader = MyScaleReader(myscale_host=host, username=username, password=password)
reader.load_data([random.random() for _ in range(1536)])

reader.load_data(
    [random.random() for _ in range(1536)],
    where_str="extra_info._dummy=0",
    limit=3,
)

logger.info("\n\n[DONE]", bright=True)