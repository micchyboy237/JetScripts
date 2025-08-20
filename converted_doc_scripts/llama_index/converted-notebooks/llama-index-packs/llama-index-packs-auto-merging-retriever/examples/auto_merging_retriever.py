from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SimpleDirectoryReader
from llama_index.core.llama_pack import download_llama_pack
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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
# Auto Merging Retriever Pack

This LlamaPack provides an example of our auto-merging retriever.

### Setup Data
"""
logger.info("# Auto Merging Retriever Pack")

# !wget "https://www.dropbox.com/s/f6bmb19xdg0xedm/paul_graham_essay.txt?dl=1" -O paul_graham_essay.txt


reader = SimpleDirectoryReader(input_files=["paul_graham_essay.txt"])
documents = reader.load_data()

"""
### Download and Initialize Pack
"""
logger.info("### Download and Initialize Pack")


AutoMergingRetrieverPack = download_llama_pack(
    "AutoMergingRetrieverPack",
    "./auto_merging_retriever_pack",
)

auto_merging_pack = AutoMergingRetrieverPack(documents)

"""
### Run Pack
"""
logger.info("### Run Pack")

response = auto_merging_pack.run("What did the author do during his time in YC?")

logger.debug(str(response))

len(response.source_nodes)

"""
### Inspect Modules
"""
logger.info("### Inspect Modules")

modules = auto_merging_pack.get_modules()
display(modules)

node_parser = auto_merging_pack.node_parser

retriever = auto_merging_pack.retriever

query_engine = auto_merging_pack.query_engine

logger.info("\n\n[DONE]", bright=True)