import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from llama_index.tools.gmail.base import GmailToolSpec
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")


# os.environ["OPENAI_API_KEY"] = "sk-your-key"



tool_spec = GmailToolSpec()

agent = FunctionAgent(
    tools=tool_spec.to_tool_list(),
    llm=MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats"),
)

ctx = Context(agent)

await agent.run(
    "Can you create a new email to helpdesk and support @example.com about a service"
    " outage",
    ctx=ctx,
)

async def run_async_code_a7f22e7d():
    await agent.run("Update the draft so that it's the same but from 'Adam'", ctx=ctx)
    return 
 = asyncio.run(run_async_code_a7f22e7d())
logger.success(format_json())

async def run_async_code_876bdfbf():
    await agent.run("display the draft", ctx=ctx)
    return 
 = asyncio.run(run_async_code_876bdfbf())
logger.success(format_json())

async def run_async_code_fdbe3122():
    await agent.run("send the draft email", ctx=ctx)
    return 
 = asyncio.run(run_async_code_fdbe3122())
logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)