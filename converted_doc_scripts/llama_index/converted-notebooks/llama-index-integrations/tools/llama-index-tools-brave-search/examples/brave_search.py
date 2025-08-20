import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.tools.brave_search.base import BraveSearchToolSpec
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")


# os.environ["OPENAI_API_KEY"] = "sk-xxx"



brave_tool = BraveSearchToolSpec(api_key="your-api-key")
agent = FunctionAgent(
    tools=brave_tool.to_tool_list(),
    llm=MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats"),
)

async def run_async_code_2abf0d8a():
    logger.debug(await agent.run("whats the latest news about superconductors"))
    return 
 = asyncio.run(run_async_code_2abf0d8a())
logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)