import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.tools.wolfram_alpha.base import WolframAlphaToolSpec
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



# os.environ["OPENAI_API_KEY"] = "sk-your-key"


wolfram_spec = WolframAlphaToolSpec(app_id="your-key")
tools = wolfram_spec.to_tool_list()


agent = FunctionAgent(
    tools=wolfram_spec.to_tool_list(),
    llm=MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats"),
)

async def run_async_code_5a6c0b69():
    logger.debug(await agent.run("what is 100000 * 12312 * 123 + 123"))
    return 
 = asyncio.run(run_async_code_5a6c0b69())
logger.success(format_json())

async def run_async_code_73e0e539():
    logger.debug(await agent.run("how many calories are in 100g of milk chocolate"))
    return 
 = asyncio.run(run_async_code_73e0e539())
logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)