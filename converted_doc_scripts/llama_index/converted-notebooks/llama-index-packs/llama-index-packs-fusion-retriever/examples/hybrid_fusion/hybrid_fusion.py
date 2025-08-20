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
# Hybrid Fusion Retriever Pack

This LlamaPack provides an example of our hybrid fusion retriever pack.
"""
logger.info("# Hybrid Fusion Retriever Pack")

# !pip install llama-index llama-hub rank-bm25

# import nest_asyncio

# nest_asyncio.apply()

"""
### Setup Data
"""
logger.info("### Setup Data")

# !wget "https://www.dropbox.com/s/f6bmb19xdg0xedm/paul_graham_essay.txt?dl=1" -O paul_graham_essay.txt


reader = SimpleDirectoryReader(input_files=["paul_graham_essay.txt"])
documents = reader.load_data()

node_parser = SimpleNodeParser.from_defaults()
nodes = node_parser.get_nodes_from_documents(documents)

"""
### Download and Initialize Pack
"""
logger.info("### Download and Initialize Pack")


HybridFusionRetrieverPack = download_llama_pack(
    "HybridFusionRetrieverPack",
    "./hybrid_fusion_pack",
)

hybrid_fusion_pack = HybridFusionRetrieverPack(
    nodes, chunk_size=256, vector_similarity_top_k=2, bm25_similarity_top_k=2
)

"""
### Run Pack
"""
logger.info("### Run Pack")

response = hybrid_fusion_pack.run("What did the author do during his time in YC?")

logger.debug(str(response))

len(response.source_nodes)

"""
### Inspect Modules
"""
logger.info("### Inspect Modules")

modules = hybrid_fusion_pack.get_modules()
display(modules)

logger.info("\n\n[DONE]", bright=True)