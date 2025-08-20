import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.agent import FunctionAgent
from llama_index.core.llms import ChatMessage
from llama_index.core.settings import Settings
from llama_index.core.tools import FunctionTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ai21 import AI21
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/ai21.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# AI21

This notebook shows how to use AI21's foundation models in LlamaIndex. The default model is `jamba-1.5-mini`.
Other supported models are `jamba-1.5-large` and `jamba-instruct`. If you want to use the older Jurassic models, specify the model name `j2-mid` or `j2-ultra`.

## Basic Usage

If you're opening this Notebook on colab, you probably need to install LlamaIndex 🦙.
"""
logger.info("# AI21")

# %pip install llama-index-llms-ai21

# !pip install llama-index

"""
## Setting the AI21 API Key

When creating an `AI21` instance, you can pass the API key as a parameter. If not provided as a parameter, it defaults to the value of the environment variable `AI21_API_KEY`.
"""
logger.info("## Setting the AI21 API Key")


api_key = <YOUR API KEY>
os.environ["AI21_API_KEY"] = api_key

llm = AI21()

llm = AI21(api_key=api_key)

"""
#### Call `chat` with a list of messages

Messages must be listed from oldest to newest, starting with a `user` role message and alternating between `user` and `assistant` messages.
"""
logger.info("#### Call `chat` with a list of messages")


messages = [
    ChatMessage(role="user", content="hello there"),
    ChatMessage(
        role="assistant", content="Arrrr, matey! How can I help ye today?"
    ),
    ChatMessage(role="user", content="What is your name?"),
]

resp = AI21(api_key=api_key).chat(
    messages, preamble_override="You are a pirate with a colorful personality"
)

logger.debug(resp)

"""
#### Call `complete` with a prompt
"""
logger.info("#### Call `complete` with a prompt")


api_key = "Your api key"
resp = AI21(api_key=api_key).complete("Paul Graham is ")

logger.debug(resp)

"""
## Call Async Methods
"""
logger.info("## Call Async Methods")


prompt = "What is the meaning of life?"

messages = [
    ChatMessage(role="user", content=prompt),
]

async def run_async_code_ae63499a():
    async def run_async_code_554d2598():
        chat_resp = AI21(api_key=api_key).chat(messages)
        return chat_resp
    chat_resp = asyncio.run(run_async_code_554d2598())
    logger.success(format_json(chat_resp))
    return chat_resp
chat_resp = asyncio.run(run_async_code_ae63499a())
logger.success(format_json(chat_resp))

async def run_async_code_65ec21a8():
    async def run_async_code_db2a939b():
        complete_resp = AI21(api_key=api_key).complete(prompt)
        return complete_resp
    complete_resp = asyncio.run(run_async_code_db2a939b())
    logger.success(format_json(complete_resp))
    return complete_resp
complete_resp = asyncio.run(run_async_code_65ec21a8())
logger.success(format_json(complete_resp))

"""
## Adjust the model behavior

Configure parameters passed to the model to adjust its behavior. For instance, setting a lower `temperature` will cause less variation between calls. Setting `temperature=0` will generate the same answer to the same question every time.
"""
logger.info("## Adjust the model behavior")


llm = AI21(
    model="jamba-1.5-mini", api_key=api_key, max_tokens=100, temperature=0.5
)

resp = llm.complete("Paul Graham is ")

logger.debug(resp)

"""
## Streaming

Stream generated responses at one token per message using the `stream_chat` method.
"""
logger.info("## Streaming")


llm = AI21(api_key=api_key, model="jamba-1.5-mini")
messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="Tell me a story"),
]
resp = llm.stream_chat(messages)

for r in resp:
    logger.debug(r.delta, end="")

"""
## Tokenizer

Different models use different tokenizers.
"""
logger.info("## Tokenizer")


llm = AI21(api_key=api_key, model="jamba-1.5-mini")

tokenizer = llm.tokenizer

tokens = tokenizer.encode("Hello llama-index!")

decoded = tokenizer.decode(tokens)

logger.debug(decoded)

"""
## Tool Calling
"""
logger.info("## Tool Calling")



def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b


def subtract(a: int, b: int) -> int:
    """Subtract two integers and returns the result integer"""
    return a - b


def divide(a: int, b: int) -> float:
    """Divide two integers and returns the result float"""
    return a - b


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)
subtract_tool = FunctionTool.from_defaults(fn=subtract)
divide_tool = FunctionTool.from_defaults(fn=divide)

llm = AI21(model="jamba-1.5-mini", api_key=api_key)

agent = FunctionAgent(
    tools=[multiply_tool, add_tool, subtract_tool, divide_tool],
    llm=llm,
)

async def async_func_37():
    response = await agent.run(
        "My friend Moses had 10 apples. He ate 5 apples in the morning. Then he found a box with 25 apples. He divided all his apples between his 5 friends. How many apples did each friend get?"
    )
    return response
response = asyncio.run(async_func_37())
logger.success(format_json(response))

logger.info("\n\n[DONE]", bright=True)