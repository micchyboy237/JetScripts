import asyncio
from jet.transformers.formatters import format_json
from IPython.display import clear_output
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.llms import ChatMessage
from llama_index.core.prompts import PromptTemplate
from llama_index.core.settings import Settings
from llama_index.core.tools import FunctionTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.mistralai import MistralAI
from pprint import pprint
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/mistralai.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# MistralAI

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# MistralAI")

# %pip install llama-index-llms-mistralai

# !pip install llama-index

"""
#### Call `complete` with a prompt
"""
logger.info("#### Call `complete` with a prompt")



llm = MistralAI(api_key="<replace-with-your-key>")

resp = llm.complete("Paul Graham is ")

logger.debug(resp)

"""
#### Call `chat` with a list of messages
"""
logger.info("#### Call `chat` with a list of messages")


messages = [
    ChatMessage(role="system", content="You are CEO of MistralAI."),
    ChatMessage(role="user", content="Tell me the story about La plateforme"),
]
resp = MistralAI().chat(messages)

logger.debug(resp)

"""
#### Call with `random_seed`
"""
logger.info("#### Call with `random_seed`")


messages = [
    ChatMessage(role="system", content="You are CEO of MistralAI."),
    ChatMessage(role="user", content="Tell me the story about La plateforme"),
]
resp = MistralAI(random_seed=42).chat(messages)

logger.debug(resp)

"""
## Streaming

Using `stream_complete` endpoint
"""
logger.info("## Streaming")


llm = MistralAI()
resp = llm.stream_complete("Paul Graham is ")

for r in resp:
    logger.debug(r.delta, end="")


llm = MistralAI()
messages = [
    ChatMessage(role="system", content="You are CEO of MistralAI."),
    ChatMessage(role="user", content="Tell me the story about La plateforme"),
]
resp = llm.stream_chat(messages)

for r in resp:
    logger.debug(r.delta, end="")

"""
## Configure Model
"""
logger.info("## Configure Model")


llm = MistralAI(model="mistral-medium")

resp = llm.stream_complete("Paul Graham is ")

for r in resp:
    logger.debug(r.delta, end="")

"""
## Function Calling

`mistral-large` supports native function calling. There's a seamless integration with LlamaIndex tools, through the `predict_and_call` function on the `llm`. 

This allows the user to attach any tools and let the LLM decide which tools to call (if any).

If you wish to perform tool calling as part of an agentic loop, check out our [agent guides](https://docs.llamaindex.ai/en/latest/module_guides/deploying/agents/) instead.

**NOTE**: If you use another Mistral model, we will use a ReAct prompt to attempt to call the function. Your mileage may vary.
"""
logger.info("## Function Calling")



def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


def mystery(a: int, b: int) -> int:
    """Mystery function on two integers."""
    return a * b + a + b


mystery_tool = FunctionTool.from_defaults(fn=mystery)
multiply_tool = FunctionTool.from_defaults(fn=multiply)

llm = MistralAI(model="mistral-large-latest")

response = llm.predict_and_call(
    [mystery_tool, multiply_tool],
    user_msg="What happens if I run the mystery function on 5 and 7",
)

logger.debug(str(response))

response = llm.predict_and_call(
    [mystery_tool, multiply_tool],
    user_msg=(
        """What happens if I run the mystery function on the following pairs of numbers? Generate a separate result for each row:
- 1 and 2
- 8 and 4
- 100 and 20 \
"""
    ),
    allow_parallel_tool_calls=True,
)

logger.debug(str(response))

for s in response.sources:
    logger.debug(f"Name: {s.tool_name}, Input: {s.raw_input}, Output: {str(s)}")

"""
You get the same result if you use the `async` variant (it will be faster since we do asyncio.gather under the hood).
"""
logger.info("You get the same result if you use the `async` variant (it will be faster since we do asyncio.gather under the hood).")

async def async_func_0():
    response = llm.predict_and_call(
        [mystery_tool, multiply_tool],
        user_msg=(
            """What happens if I run the mystery function on the following pairs of numbers? Generate a separate result for each row:
    - 1 and 2
    - 8 and 4
    - 100 and 20 \
    """
        ),
        allow_parallel_tool_calls=True,
    )
    return response
response = asyncio.run(async_func_0())
logger.success(format_json(response))
for s in response.sources:
    logger.debug(f"Name: {s.tool_name}, Input: {s.raw_input}, Output: {str(s)}")

"""
## Structured Prediction

An important use case for function calling is extracting structured objects. LlamaIndex provides an intuitive interface for converting any LLM into a structured LLM - simply define the target Pydantic class (can be nested), and given a prompt, we extract out the desired object.
"""
logger.info("## Structured Prediction")



class Restaurant(BaseModel):
    """A restaurant with name, city, and cuisine."""

    name: str
    city: str
    cuisine: str


llm = MistralAI(model="mistral-large-latest")
prompt_tmpl = PromptTemplate(
    "Generate a restaurant in a given city {city_name}"
)

restaurant_obj = (
    llm.as_structured_llm(Restaurant)
    .complete(prompt_tmpl.format(city_name="Miami"))
    .raw
)

restaurant_obj

"""
#### Structured Prediction with Streaming

Any LLM wrapped with `as_structured_llm` supports streaming through `stream_chat`.
"""
logger.info("#### Structured Prediction with Streaming")


input_msg = ChatMessage.from_str("Generate a restaurant in Miami")

sllm = llm.as_structured_llm(Restaurant)
stream_output = sllm.stream_chat([input_msg])
for partial_output in stream_output:
    clear_output(wait=True)
    plogger.debug(partial_output.raw.dict())
    restaurant_obj = partial_output.raw

restaurant_obj

"""
## Async
"""
logger.info("## Async")


llm = MistralAI()
async def run_async_code_c3ecd675():
    async def run_async_code_a989c387():
        resp = llm.complete("Paul Graham is ")
        return resp
    resp = asyncio.run(run_async_code_a989c387())
    logger.success(format_json(resp))
    return resp
resp = asyncio.run(run_async_code_c3ecd675())
logger.success(format_json(resp))

logger.debug(resp)

logger.info("\n\n[DONE]", bright=True)