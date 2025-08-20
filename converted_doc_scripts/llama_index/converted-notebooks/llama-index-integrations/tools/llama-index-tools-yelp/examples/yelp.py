import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from llama_index.tools.tool_spec.load_and_search.base import LoadAndSearchToolSpec
from llama_index.tools.yelp.base import YelpToolSpec
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")


# os.environ["OPENAI_API_KEY"] = "sk-your-key"


tool_spec = YelpToolSpec(api_key="your-key", client_id="your-id")


tools = tool_spec.to_tool_list()
agent = FunctionAgent(
    tools=[
        *LoadAndSearchToolSpec.from_defaults(tools[0]).to_tool_list(),
        *LoadAndSearchToolSpec.from_defaults(tools[1]).to_tool_list(),
    ],
    llm=MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats"),
)


ctx = Context(agent)

async def run_async_code_1cd07748():
    logger.debug(await agent.run("what good resturants are in toronto", ctx=ctx))
    return 
 = asyncio.run(run_async_code_1cd07748())
logger.success(format_json())
async def run_async_code_43137992():
    logger.debug(await agent.run("what are the details of lao lao bar", ctx=ctx))
    return 
 = asyncio.run(run_async_code_43137992())
logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)