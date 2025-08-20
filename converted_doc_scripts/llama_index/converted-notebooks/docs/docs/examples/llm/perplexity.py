import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool
from llama_index.llms.perplexity import Perplexity
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Perplexity

Perplexity's Sonar API offers a solution that combines real-time, grounded web search with advanced reasoning and deep research capabilities. 

When to use:

- When your application requires timely, relevant data directly from the web, such as dynamic content updates or current event tracking.
- For products that need to support complex user queries with integrated reasoning and deep research, like digital assistants or advanced search engines.

Before we get started, make sure you install `llama_index`
"""
logger.info("# Perplexity")

# %pip install llama-index-llms-perplexity

# !pip install llama-index

"""
## Initial Setup

As of April 12th, 2025 - the following models are supported with the Perplexity LLM class in LLaMa Index:

| Model                | Context Length | Model Type      |
|----------------------|----------------|-----------------|
| `sonar-deep-research`  | 128k           | Chat Completion |
| `sonar-reasoning-pro`  | 128k           | Chat Completion |
| `sonar-reasoning`      | 128k           | Chat Completion |
| `sonar-pro`            | 200k           | Chat Completion |
| `sonar`                | 128k           | Chat Completion |
| `r1-1776`              | 128k           | Chat Completion |

- `sonar-pro` has a max output token limit of 8k.
- The reasoning models output Chain of Thought responses.
- `r1-1776` is an offline chat model that does not use the Perplexity search subsystem.



You can find the latest supported models [here](https://docs.perplexity.ai/docs/model-cards) \
Rate limits are found [here](https://docs.perplexity.ai/docs/rate-limits) \
Pricing can be found [here](https://docs.perplexity.ai/guides/pricing).
"""
logger.info("## Initial Setup")

# import getpass

if "PPLX_API_KEY" not in os.environ:
#     os.environ["PPLX_API_KEY"] = getpass.getpass(
        "Enter your Perplexity API key: "
    )


PPLX_API_KEY = __import__("os").environ.get("PPLX_API_KEY")

llm = Perplexity(api_key=PPLX_API_KEY, model="sonar-pro", temperature=0.2)


messages_dict = [
    {"role": "system", "content": "Be precise and concise."},
    {
        "role": "user",
        "content": "Tell me the latest news about the US Stock Market.",
    },
]

messages = [ChatMessage(**msg) for msg in messages_dict]

logger.debug(messages)

"""
## Chat
"""
logger.info("## Chat")

response = llm.chat(messages)
logger.debug(response)

"""
## Async Chat

For asynchronous conversation processing, use the `chat` method to send messages and the response:
"""
logger.info("## Async Chat")

async def run_async_code_d8dceed1():
    async def run_async_code_c21ac29f():
        response = llm.chat(messages)
        return response
    response = asyncio.run(run_async_code_c21ac29f())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_d8dceed1())
logger.success(format_json(response))

logger.debug(response)

"""
## Stream Chat

For cases where you want to receive a response token by token in real time, use the `stream_chat` method:
"""
logger.info("## Stream Chat")

response = llm.stream_chat(messages)

for r in response:
    logger.debug(r.delta, end="")

"""
## Async Stream Chat

Similarly, for asynchronous streaming, the `astream_chat` method provides a way to process response deltas asynchronously:
"""
logger.info("## Async Stream Chat")

async def run_async_code_548f10f2():
    async def run_async_code_0bc31564():
        resp = llm.stream_chat(messages)
        return resp
    resp = asyncio.run(run_async_code_0bc31564())
    logger.success(format_json(resp))
    return resp
resp = asyncio.run(run_async_code_548f10f2())
logger.success(format_json(resp))

async for delta in resp:
    logger.debug(delta.delta, end="")

"""
### Tool calling 

Perplexity models can easily be wrapped into a llamaindex tool so that it can be called as part of your data processing or conversational workflows. This tool uses real-time generative search powered by Perplexity, and it’s configured with the updated default model ("sonar-pro") and the enable_search_classifier parameter enabled.

Below is an example of how to define and register the tool:
"""
logger.info("### Tool calling")



def query_perplexity(query: str) -> str:
    """
    Queries the Perplexity API via the LlamaIndex integration.

    This function instantiates a Perplexity LLM with updated default settings
    (using model "sonar-pro" and enabling search classifier so that the API can
    intelligently decide if a search is needed), wraps the query into a ChatMessage,
    and returns the generated response content.
    """
    pplx_api_key = (
        "your-perplexity-api-key"  # Replace with your actual API key
    )

    llm = Perplexity(
        api_key=pplx_api_key,
        model="sonar-pro",
        temperature=0.7,
        enable_search_classifier=True,  # This will determine if the search component is necessary in this particular context
    )

    messages = [ChatMessage(role="user", content=query)]
    response = llm.chat(messages)
    return response.message.content


query_perplexity_tool = FunctionTool.from_defaults(fn=query_perplexity)

logger.info("\n\n[DONE]", bright=True)