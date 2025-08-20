from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llama_pack import download_llama_pack
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.packs.recursive_retriever import (
EmbeddedTablesUnstructuredRetrieverPack,
)
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
# Embedded Tables Pack

This LlamaPack provides an example of our embedded-tables pack (with recursive retrieval + Unstructured.io).
"""
logger.info("# Embedded Tables Pack")

# %pip install llama-index-packs-recursive-retriever

# !pip install llama-index llama-hub unstructured==0.10.18 lxml

# import nest_asyncio

# nest_asyncio.apply()

"""
### Setup Data
"""
logger.info("### Setup Data")

# !wget "https://www.dropbox.com/scl/fi/mlaymdy1ni1ovyeykhhuk/tesla_2021_10k.htm?rlkey=qf9k4zn0ejrbm716j0gg7r802&dl=1" -O tesla_2021_10k.htm

"""
### Download and Initialize Pack

Note that this pack directly takes in the html file, no need to load it beforehand.
"""
logger.info("### Download and Initialize Pack")


EmbeddedTablesUnstructuredRetrieverPack = download_llama_pack(
    "EmbeddedTablesUnstructuredRetrieverPack",
    "./embedded_tables_unstructured_pack",
)


embedded_tables_unstructured_pack = EmbeddedTablesUnstructuredRetrieverPack(
    "tesla_2021_10k.htm", nodes_save_path="2021_nodes.pkl"
)

"""
### Run Pack
"""
logger.info("### Run Pack")

response = embedded_tables_unstructured_pack.run("What was the revenue in 2020?")

logger.debug(str(response))

len(response.source_nodes)

"""
### Inspect Modules
"""
logger.info("### Inspect Modules")

modules = embedded_tables_unstructured_pack.get_modules()
display(modules)

logger.info("\n\n[DONE]", bright=True)