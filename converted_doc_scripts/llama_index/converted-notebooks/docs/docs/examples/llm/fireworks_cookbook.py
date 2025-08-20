import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.base.utils import to_openai_tool
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.settings import Settings
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.fireworks import Fireworks
from llama_index.program.openai import MLXPydanticProgram
from pydantic import BaseModel
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/fireworks_cookbook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Fireworks Function Calling Cookbook

Fireworks.ai supports function calling for its LLMs, similar to MLX. This lets users directly describe the set of tools/functions available and have the model dynamically pick the right function calls to invoke, without complex prompting on the user's part.

Since our Fireworks LLM directly subclasses MLX, we can use our existing abstractions with Fireworks.

We show this on three levels: directly on the model API, as part of a Pydantic Program (structured output extraction), and as part of an agent.
"""
logger.info("# Fireworks Function Calling Cookbook")

# %pip install llama-index-llms-fireworks

# %pip install llama-index


os.environ["FIREWORKS_API_KEY"] = ""


llm = Fireworks(
    model="accounts/fireworks/models/firefunction-v1", temperature=0
)

"""
## Function Calling on the LLM Module

You can directly input function calls on the LLM module.
"""
logger.info("## Function Calling on the LLM Module")



class Song(BaseModel):
    """A song with name and artist"""

    name: str
    artist: str


song_fn = to_openai_tool(Song)


response = llm.complete("Generate a song from Beyonce", tools=[song_fn])
tool_calls = response.additional_kwargs["tool_calls"]
logger.debug(tool_calls)

"""
## Using a Pydantic Program

Our Pydantic programs allow structured output extraction into a Pydantic object. `MLXPydanticProgram` takes advantage of function calling for structured output extraction.
"""
logger.info("## Using a Pydantic Program")


prompt_template_str = "Generate a song about {artist_name}"
program = MLXPydanticProgram.from_defaults(
    output_cls=Song, prompt_template_str=prompt_template_str, llm=llm
)

output = program(artist_name="Eminem")

output

"""
## Using An MLX Agent
"""
logger.info("## Using An MLX Agent")




def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


add_tool = FunctionTool.from_defaults(fn=add)

agent = FunctionAgent(
    tools=[multiply_tool, add_tool],
    llm=llm,
)

async def run_async_code_efa6c44f():
    async def run_async_code_cddfcfeb():
        response = await agent.run("What is (121 * 3) + 42?")
        return response
    response = asyncio.run(run_async_code_cddfcfeb())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_efa6c44f())
logger.success(format_json(response))
logger.debug(str(response))

logger.info("\n\n[DONE]", bright=True)