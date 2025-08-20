import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from llama_index.tools.tool_spec.load_and_search.base import LoadAndSearchToolSpec
from llama_index.tools.wikipedia.base import WikipediaToolSpec
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")


# os.environ["OPENAI_API_KEY"] = "sk-your-key"


wiki_spec = WikipediaToolSpec()
tool = wiki_spec.to_tool_list()[1]


agent = FunctionAgent(
    tools=LoadAndSearchToolSpec.from_defaults(tool).to_tool_list(),
    llm=MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats"),
)


ctx = Context(agent)

async def run_async_code_bb3f6c06():
    logger.debug(await agent.run("what is the capital of poland", ctx=ctx))
    return 
 = asyncio.run(run_async_code_bb3f6c06())
logger.success(format_json())

async def run_async_code_23a1d5bd():
    logger.debug(await agent.run("how long has poland existed", ctx=ctx))
    return 
 = asyncio.run(run_async_code_23a1d5bd())
logger.success(format_json())

async def run_async_code_5105c882():
    logger.debug(await agent.run("using information already loaded, how big is poland?", ctx=ctx))
    return 
 = asyncio.run(run_async_code_5105c882())
logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)