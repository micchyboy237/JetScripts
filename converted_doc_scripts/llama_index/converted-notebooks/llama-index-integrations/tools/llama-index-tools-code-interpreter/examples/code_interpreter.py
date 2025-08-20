import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from llama_index.tools.code_interpreter.base import CodeInterpreterToolSpec
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

# %pip install llama-index-agent-openai
# %pip install llama-index-llms-ollama
# %pip install llama-index-tools-code-interpreter

# !pip install llama-index


# os.environ["OPENAI_API_KEY"] = "sk-..."



code_spec = CodeInterpreterToolSpec()

tools = code_spec.to_tool_list()

agent = FunctionAgent(
    tools=tools,
    llm=MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats"),
)

ctx = Context(agent)

logger.debug(
    await agent.run(
        "Can you help me write some python code to pass to the code_interpreter tool",
        ctx=ctx
    )
)

logger.debug(
    await agent.run(
        """There is a world_happiness_2016.csv file in the `data` directory (relative path).
                 Can you write and execute code to tell me columns does it have?""",
        ctx=ctx,
    )
)

async def run_async_code_ea5fed18():
    logger.debug(await agent.run("What are the top 10 happiest countries", ctx=ctx))
    return 
 = asyncio.run(run_async_code_ea5fed18())
logger.success(format_json())

async def run_async_code_ea50ba74():
    logger.debug(await agent.run("Can you make a graph of the top 10 happiest countries", ctx=ctx))
    return 
 = asyncio.run(run_async_code_ea50ba74())
logger.success(format_json())

async def run_async_code_ea50ba74():
    logger.debug(await agent.run("Can you make a graph of the top 10 happiest countries", ctx=ctx))
    return 
 = asyncio.run(run_async_code_ea50ba74())
logger.success(format_json())

async def run_async_code_ee948054():
    logger.debug(await agent.run("can you also plot the 10 lowest", ctx=ctx))
    return 
 = asyncio.run(run_async_code_ee948054())
logger.success(format_json())

async def run_async_code_73b48003():
    logger.debug(await agent.run("can you do it in one plot", ctx=ctx))
    return 
 = asyncio.run(run_async_code_73b48003())
logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)