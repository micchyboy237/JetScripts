import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from llama_index.tools.text_to_image.base import TextToImageToolSpec
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")


# os.environ["OPENAI_API_KEY"] = "sk-..."



text_to_image_spec = TextToImageToolSpec()
tools = text_to_image_spec.to_tool_list()

agent = FunctionAgent(
    tools=tools,
    llm=MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats"),
)


ctx = Context(agent)

async def run_async_code_0156e5c4():
    logger.debug(await agent.run("show 2 images of a beautiful beach with a palm tree at sunset", ctx=ctx))
    return 
 = asyncio.run(run_async_code_0156e5c4())
logger.success(format_json())

async def run_async_code_283fff8b():
    logger.debug(await agent.run("make the second image higher quality", ctx=ctx))
    return 
 = asyncio.run(run_async_code_283fff8b())
logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)