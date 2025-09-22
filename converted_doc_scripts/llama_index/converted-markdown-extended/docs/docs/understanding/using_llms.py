from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import Ollama
from jet.logger import logger
from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
---
sidebar:
  order: 2
---
# Using LLMs

<Aside type="tip">
For a list of our supported LLMs and a comparison of their functionality, check out our [LLM module guide](/python/framework/module_guides/models/llms).
</Aside>

One of the first steps when building an LLM-based application is which LLM to use; they have different strengths and price points and you may wish to use more than one.

LlamaIndex provides a single interface to a large number of different LLMs. Using an LLM can be as simple as installing the appropriate integration:
"""
logger.info("# Using LLMs")

pip install llama-index-llms-ollama

"""
And then calling it in a one-liner:
"""
logger.info("And then calling it in a one-liner:")


response = Ollama().complete("William Shakespeare is ")
logger.debug(response)

"""
# Note that this requires an API key called `OPENAI_API_KEY` in your environment; see the [starter tutorial](/python/framework/getting_started/starter_example) for more details.

`complete` is also available as an async method, `acomplete`.

You can also get a streaming response by calling `stream_complete`, which returns a generator that yields tokens as they are produced:

handle = Ollama().stream_complete("William Shakespeare is ")

for token in handle:
    logger.debug(token.delta, end="", flush=True)

`stream_complete` is also available as an async method, `astream_complete`.

## Chat interface

The LLM class also implements a `chat` method, which allows you to have more sophisticated interactions:
"""
logger.info("## Chat interface")

messages = [
    ChatMessage(role="system", content="You are a helpful assistant."),
    ChatMessage(role="user", content="Tell me a joke."),
]
chat_response = llm.chat(messages)

"""
`stream_chat` and `astream_chat` are also available.

## Specifying models

Many LLM integrations provide more than one model. You can specify a model by passing the `model` parameter to the LLM constructor:
"""
logger.info("## Specifying models")

llm = Ollama(model="llama3.2")
response = llm.complete("Who is Laurie Voss?")
logger.debug(response)

"""
## Multi-Modal LLMs

Some LLMs support multi-modal chat messages. This means that you can pass in a mix of text and other modalities (images, audio, video, etc.) and the LLM will handle it.

Currently, LlamaIndex supports text, images, and audio inside ChatMessages using content blocks.
"""
logger.info("## Multi-Modal LLMs")


llm = Ollama(model="llama3.2")

messages = [
    ChatMessage(
        role="user",
        blocks=[
            ImageBlock(path="image.png"),
            TextBlock(text="Describe the image in a few sentences."),
        ],
    )
]

resp = llm.chat(messages)
logger.debug(resp.message.content)

"""
## Tool Calling

Some LLMs (Ollama, Ollama, Gemini, Ollama, etc.) support tool calling directly over API calls -- this means tools and functions can be called without specific prompts and parsing mechanisms.
"""
logger.info("## Tool Calling")



def generate_song(name: str, artist: str) -> Song:
    """Generates a song with provided name and artist."""
    return {"name": name, "artist": artist}


tool = FunctionTool.from_defaults(fn=generate_song)

llm = Ollama(model="llama3.2")
response = llm.predict_and_call(
    [tool],
    "Pick a random song for me",
)
logger.debug(str(response))

"""
For more details on even more advanced tool calling, check out the in-depth guide using [Ollama](/python/examples/llm/ollama). The same approaches work for any LLM that supports tools/functions (e.g. Ollama, Gemini, Ollama, etc.).

You can learn more about tools and agents in the [tools guide](/python/framework/understanding/agent/tools).

## Available LLMs

We support integrations with Ollama, Ollama, Mistral, DeepSeek, Hugging Face, and dozens more. Check out our [module guide to LLMs](/python/framework/module_guides/models/llms) for a full list, including how to run a local model.

<Aside type="tip">
A general note on privacy and LLM usage can be found on the [privacy page](/python/framework/understanding/privacy).
</Aside>

### Using a local LLM

LlamaIndex doesn't just support hosted LLM APIs; you can also run a local model such as Meta's Llama 3 locally. For example, if you have [Ollama](https://github.com/ollama/ollama) installed and running:
"""
logger.info("## Available LLMs")


llm = Ollama(
    model="llama3.3",
    request_timeout=60.0,
    context_window=8000,
)

"""
See the [custom LLM's How-To](/python/framework/module_guides/models/llms/usage_custom) for more details on using and configuring LLM models.
"""
logger.info("See the [custom LLM's How-To](/python/framework/module_guides/models/llms/usage_custom) for more details on using and configuring LLM models.")

logger.info("\n\n[DONE]", bright=True)