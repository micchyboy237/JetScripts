from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SimpleDirectoryReader
from llama_index.core.llama_pack import download_llama_pack
from llama_index.core.node_parser import SimpleNodeParser
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
# Small-to-big Retrieval Pack

This LlamaPack provides an example of our small-to-big retrieval (with recursive retrieval).
"""
logger.info("# Small-to-big Retrieval Pack")

# import nest_asyncio

# nest_asyncio.apply()

"""
### Setup Data
"""
logger.info("### Setup Data")

# !wget "https://www.dropbox.com/s/f6bmb19xdg0xedm/paul_graham_essay.txt?dl=1" -O paul_graham_essay.txt


reader = SimpleDirectoryReader(input_files=["paul_graham_essay.txt"])
documents = reader.load_data()

"""
### Download and Initialize Pack

Note that this pack directly takes in the html file, no need to load it beforehand.
"""
logger.info("### Download and Initialize Pack")


RecursiveRetrieverSmallToBigPack = download_llama_pack(
    "RecursiveRetrieverSmallToBigPack",
    "./recursive_retriever_stb_pack",
)

recursive_retriever_stb_pack = RecursiveRetrieverSmallToBigPack(
    documents,
)

"""
### Run Pack
"""
logger.info("### Run Pack")

response = recursive_retriever_stb_pack.run("What did the author do growing up?")

logger.debug(str(response))

len(response.source_nodes)

"""
### Inspect Modules
"""
logger.info("### Inspect Modules")

modules = recursive_retriever_stb_pack.get_modules()
display(modules)

logger.info("\n\n[DONE]", bright=True)