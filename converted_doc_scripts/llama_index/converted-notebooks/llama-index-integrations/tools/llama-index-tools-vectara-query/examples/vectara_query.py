import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.tools.vectara_query.base import VectaraQueryToolSpec
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-vectara-query/examples/vectara_query.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Vectara Query Tool

Please note that this example notebook is only for Vectara Query tool versions >=0.3.0

To get started with Vectara, [sign up](https://vectara.com/integrations/llamaindex) (if you haven't already) and follow our [quickstart](https://docs.vectara.com/docs/quickstart) guide to create a corpus and an API key.

Once you have done this, add the following variables to your environment:

`VECTARA_CORPUS_KEY`: The corpus key for the Vectara corpus that you want your tool to search for information.

`VECTARA_API_KEY`: An API key that can perform queries on this corpus.

You are now ready to use the Vectara query tool.

To initialize the tool, provide your Vectara information and any query parameters that you want to adjust, such as the reranker, summarizer prompt, etc. To see the entire list of parameters, see the [VectaraQueryToolSpec class definition](https://github.com/run-llama/llama_index/blob/05828d6d099e78df51c76b8c98aa3ecbd45162ec/llama-index-integrations/tools/llama-index-tools-vectara-query/llama_index/tools/vectara_query/base.py#L11).
"""
logger.info("## Vectara Query Tool")


tool_spec = VectaraQueryToolSpec()

"""
After initializing the tool spec, we can provide it to our agent. For this notebook, we will use the MLX Agent, but our tool can be used with any type of agent. You will need your own MLX API key to run this notebook.
"""
logger.info("After initializing the tool spec, we can provide it to our agent. For this notebook, we will use the MLX Agent, but our tool can be used with any type of agent. You will need your own MLX API key to run this notebook.")


agent = FunctionAgent(
    tools=tool_spec.to_tool_list(),
    llm=MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats"),
)

async def run_async_code_abf115a8():
    logger.debug(await agent.run("What are the different types of electric vehicles?"))
    return 
 = asyncio.run(run_async_code_abf115a8())
logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)