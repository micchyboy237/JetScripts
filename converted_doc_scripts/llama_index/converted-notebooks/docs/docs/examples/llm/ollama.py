import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.llms import ChatMessage
from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
from llama_index.llms.ollama import Ollama
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/ollama.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Ollama LLM

## Setup
First, follow the [readme](https://github.com/jmorganca/ollama) to set up and run a local Ollama instance.

When the Ollama app is running on your local machine:
- All of your local models are automatically served on localhost:11434
- Select your model when setting llm = Ollama(..., model="<model family>:<version>")
- Increase defaullt timeout (30 seconds) if needed setting Ollama(..., request_timeout=300.0)
- If you set llm = Ollama(..., model="<model family") without a version it will simply look for latest
- By default, the maximum context window for your model is used. You can manually set the `context_window` to limit memory usage.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Ollama LLM")

# %pip install llama-index-llms-ollama


llm = Ollama(
    model="llama3.1:latest",
    request_timeout=120.0,
    context_window=8000,
)

resp = llm.complete("Who is Paul Graham?")

logger.debug(resp)

"""
#### Call `chat` with a list of messages
"""
logger.info("#### Call `chat` with a list of messages")


messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.chat(messages)

logger.debug(resp)

"""
### Streaming

Using `stream_complete` endpoint
"""
logger.info("### Streaming")

response = llm.stream_complete("Who is Paul Graham?")

for r in response:
    logger.debug(r.delta, end="")

"""
Using `stream_chat` endpoint
"""
logger.info("Using `stream_chat` endpoint")


messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.stream_chat(messages)

for r in resp:
    logger.debug(r.delta, end="")

"""
## JSON Mode

Ollama also supports a JSON mode, which tries to ensure all responses are valid JSON.

This is particularly useful when trying to run tools that need to parse structured outputs.
"""
logger.info("## JSON Mode")

llm = Ollama(
    model="llama3.1:latest",
    request_timeout=120.0,
    json_mode=True,
    context_window=8000,
)

response = llm.complete(
    "Who is Paul Graham? Output as a structured JSON object."
)
logger.debug(str(response))

"""
## Structured Outputs

We can also attach a pyndatic class to the LLM to ensure structured outputs. This will use Ollama's builtin structured output capabilities for a given pydantic class.
"""
logger.info("## Structured Outputs")



class Song(BaseModel):
    """A song with name and artist."""

    name: str
    artist: str

llm = Ollama(
    model="llama3.1:latest",
    request_timeout=120.0,
    context_window=8000,
)

sllm = llm.as_structured_llm(Song)


response = sllm.chat([ChatMessage(role="user", content="Name a random song!")])
logger.debug(response.message.content)

"""
Or with async
"""
logger.info("Or with async")

async def async_func_0():
    response = sllm.chat(
        [ChatMessage(role="user", content="Name a random song!")]
    )
    return response
response = asyncio.run(async_func_0())
logger.success(format_json(response))
logger.debug(response.message.content)

"""
You can also stream structured outputs! Streaming a structured output is a little different than streaming a normal string. It will yield a generator of the most up to date structured object.
"""
logger.info("You can also stream structured outputs! Streaming a structured output is a little different than streaming a normal string. It will yield a generator of the most up to date structured object.")

response_gen = sllm.stream_chat(
    [ChatMessage(role="user", content="Name a random song!")]
)
for r in response_gen:
    logger.debug(r.message.content)

"""
## Multi-Modal Support

Ollama supports multi-modal models, and the Ollama LLM class natively supports images out of the box.

This leverages the content blocks feature of the chat messages.

Here, we leverage the `llama3.2-vision` model to answer a question about an image. If you don't have this model yet, you'll want to run `ollama pull llama3.2-vision`.
"""
logger.info("## Multi-Modal Support")

# !wget "https://pbs.twimg.com/media/GVhGD1PXkAANfPV?format=jpg&name=4096x4096" -O ollama_image.jpg


llm = Ollama(
    model="llama3.2-vision",
    request_timeout=120.0,
    context_window=8000,
)

messages = [
    ChatMessage(
        role="user",
        blocks=[
            TextBlock(text="What type of animal is this?"),
            ImageBlock(path="ollama_image.jpg"),
        ],
    ),
]

resp = llm.chat(messages)
logger.debug(resp)

"""
Close enough ;)

## Thinking

Models in Ollama support "thinking" -- the process of reasoning and reflecting on a response before returning a final answer.

Below we show how to enable thinking in Ollama models in both streaming and non-streaming modes using the `thinking` parameter and the `qwen3:8b` model.
"""
logger.info("## Thinking")


llm = Ollama(
    model="qwen3:8b",
    request_timeout=360,
    thinking=True,
    context_window=8000,
)

resp = llm.complete("What is 434 / 22?")

logger.debug(resp.additional_kwargs["thinking"])

logger.debug(resp.text)

"""
Thats a lot of thinking!

Now, let's try a streaming example to make the wait less painful:
"""
logger.info("Thats a lot of thinking!")

resp_gen = llm.stream_complete("What is 434 / 22?")

thinking_started = False
response_started = False

for resp in resp_gen:
    if resp.additional_kwargs.get("thinking_delta", None):
        if not thinking_started:
            logger.debug("\n\n-------- Thinking: --------\n")
            thinking_started = True
            response_started = False
        logger.debug(resp.additional_kwargs["thinking_delta"], end="", flush=True)
    if resp.delta:
        if not response_started:
            logger.debug("\n\n-------- Response: --------\n")
            response_started = True
            thinking_started = False
        logger.debug(resp.delta, end="", flush=True)

logger.info("\n\n[DONE]", bright=True)