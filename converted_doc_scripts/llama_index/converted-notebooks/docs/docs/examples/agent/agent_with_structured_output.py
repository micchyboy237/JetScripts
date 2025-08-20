import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pydantic import BaseModel, Field
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
# Agents with Structured Outputs

When you run your agent or multi-agent framework, you might want it to output the result in a specific format. In this notebook, we will see a simple example of how to apply this to a FunctionAgent!🦙🚀

Let's first install the needed dependencies
"""
logger.info("# Agents with Structured Outputs")

# ! pip install llama-index

# from getpass import getpass

# os.environ["OPENAI_API_KEY"] = getpass()

"""
Let's now define our structured output format
"""
logger.info("Let's now define our structured output format")



class MathResult(BaseModel):
    operation: str = Field(description="The operation that has been performed")
    result: int = Field(description="Result of the operation")

"""
And a very simple calculator agent
"""
logger.info("And a very simple calculator agent")


llm = MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats")


def add(x: int, y: int):
    """Add two numbers"""
    return x + y


def multiply(x: int, y: int):
    """Multiply two numbers"""
    return x * y


agent = FunctionAgent(
    llm=llm,
    output_cls=MathResult,
    tools=[add, multiply],
    system_prompt="You are a calculator agent that can add or multiply two numbers by calling tools",
    name="calculator",
)

"""
Let's now run the agent
"""
logger.info("Let's now run the agent")

async def run_async_code_d3493c9a():
    async def run_async_code_cf2ce179():
        response = await agent.run("What is the result of 10 multiplied by 4?")
        return response
    response = asyncio.run(run_async_code_cf2ce179())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_d3493c9a())
logger.success(format_json(response))

"""
Finally, we can get the structured output
"""
logger.info("Finally, we can get the structured output")

logger.debug(response.structured_response)
logger.debug(response.get_pydantic_model(MathResult))

logger.info("\n\n[DONE]", bright=True)