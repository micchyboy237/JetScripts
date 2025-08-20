import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from langchain.chat_models import ChatAnyscale, ChatMLX
from llama_index.core import PromptTemplate
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.langchain import LangChainLLM
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/llm_predictor.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# LLM Predictor

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# LLM Predictor")

# %pip install llama-index-llms-ollama
# %pip install llama-index-llms-langchain

# !pip install llama-index

"""
## LangChain LLM
"""
logger.info("## LangChain LLM")


llm = LangChainLLM(ChatMLX())

async def run_async_code_285ba199():
    async def run_async_code_64abf00b():
        stream = llm.stream(PromptTemplate("Hi, write a short story"))
        return stream
    stream = asyncio.run(run_async_code_64abf00b())
    logger.success(format_json(stream))
    return stream
stream = asyncio.run(run_async_code_285ba199())
logger.success(format_json(stream))

async for token in stream:
    logger.debug(token, end="")

llm = LangChainLLM(ChatAnyscale())

stream = llm.stream(
    PromptTemplate("Hi, Which NFL team have most Super Bowl wins")
)
for token in stream:
    logger.debug(token, end="")

"""
## MLX LLM
"""
logger.info("## MLX LLM")


llm = MLX()

async def run_async_code_13458201():
    async def run_async_code_bcef382e():
        stream = llm.stream("Hi, write a short story")
        return stream
    stream = asyncio.run(run_async_code_bcef382e())
    logger.success(format_json(stream))
    return stream
stream = asyncio.run(run_async_code_13458201())
logger.success(format_json(stream))

for token in stream:
    logger.debug(token, end="")

logger.info("\n\n[DONE]", bright=True)