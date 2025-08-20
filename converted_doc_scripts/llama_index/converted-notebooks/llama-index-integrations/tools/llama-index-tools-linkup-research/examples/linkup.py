import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.agent import FunctionAgent
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.tools.linkup_research.base import LinkupToolSpec
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
# Building a Linkup Data Agent

This tutorial walks through using the LLM tools provided by the [Linkup API](https://app.linkup.so/) to allow LLMs to easily search and retrieve relevant content from the Internet.

To get started, you will need an [MLX api key](https://platform.openai.com/account/api-keys) and a [Linkup API key](https://app.linkup.so)

We will import the relevant agents and tools and pass them our keys here:
"""
logger.info("# Building a Linkup Data Agent")

# %pip install llama-index-tools-linkup-research llama-index


# os.environ["OPENAI_API_KEY"] = "sk-..."


linkup_tool = LinkupToolSpec(
    api_key="your Linkup API Key",
    depth="",  # Choose (standard) for a faster result (deep) for a slower but more complete result.
    output_type="",  # Choose (searchResults) for a list of results relative to your query, (sourcedAnswer) for an answer and a list of sources, or (structured) if you want a specific shema.
)


agent = FunctionAgent(
    tools=linkup_tool.to_tool_list(),
    llm=MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit"),
)

async def run_async_code_49b34829():
    logger.debug(await agent.run("Can you tell me which women were awarded the Physics Nobel Prize"))
    return 
 = asyncio.run(run_async_code_49b34829())
logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)