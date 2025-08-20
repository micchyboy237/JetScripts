from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index import GPTSimpleVectorIndex
from llama_index import download_loader
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
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
# Bored Llama: BoardDocs in LLaMA Index!

This is a fun experiment to see if we can crawl a BoardDocs site to index it for LangChain fun.
"""
logger.info("# Bored Llama: BoardDocs in LLaMA Index!")


BoardDocsReader = download_loader(
    "BoardDocsReader",
    loader_hub_url=(
        "https://raw.githubusercontent.com/dweekly/llama-hub/boarddocs/llama_hub"
    ),
    refresh_cache=True,
)
loader = BoardDocsReader(site="ca/redwood", committee_id="A4EP6J588C05")


documents = loader.load_data(meeting_ids=["CPSNV9612DF1"])

index = GPTSimpleVectorIndex.from_documents(documents)

answer = index.query("When did Trustee Weekly start attending meetings?")
logger.debug(answer.response)

logger.info("\n\n[DONE]", bright=True)