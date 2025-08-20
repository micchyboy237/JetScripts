from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SummaryIndex
from llama_index.core.response.notebook_utils import display_response
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import ImageTabularChartReader
from pathlib import Path
import os
import shutil


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
# Deplot Reader Demo

In this notebook we showcase the capabilities of our ImageTabularChartReader, which is powered by the DePlot model https://arxiv.org/abs/2212.10505.
"""
logger.info("# Deplot Reader Demo")

# %pip install llama-index-readers-file


loader = ImageTabularChartReader(keep_image=True)

"""
## Load Protected Waters Chart

This chart shows the percentage of marine territorial waters that are protected for each country.
"""
logger.info("## Load Protected Waters Chart")

documents = loader.load_data(file=Path("./marine_chart.png"))

logger.debug(documents[0].text)

summary_index = SummaryIndex.from_documents(documents)
response = summary_index.as_query_engine().query(
    "What is the difference between the shares of Greenland and the share of"
    " Mauritania?"
)

display_response(response, show_source=True)

"""
## Load Pew Research Chart

Here we load in a Pew Research chart showing international views of the US/Biden.

Source: https://www.pewresearch.org/global/2023/06/27/international-views-of-biden-and-u-s-largely-positive/
"""
logger.info("## Load Pew Research Chart")

documents = loader.load_data(file=Path("./pew1.png"))

logger.debug(documents[0].text)

summary_index = SummaryIndex.from_documents(documents)
response = summary_index.as_query_engine().query(
    "What percentage says that the US contributes to peace and stability?"
)

display_response(response, show_source=True)

logger.info("\n\n[DONE]", bright=True)