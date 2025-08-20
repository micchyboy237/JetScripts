import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.base import MLX
from jet.llm.mlx.base import MLXResponses
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llms import ChatMessage
from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock
from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
from llama_index.core.prompts import PromptTemplate
from llama_index.core.settings import Settings
from llama_index.core.tools import FunctionTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pydantic import BaseModel
from typing import List
import base64
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/openai_responses.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# MLX Responses API

This notebook shows how to use the MLX Responses LLM.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# MLX Responses API")

# %pip install llama-index llama-index-llms-ollama

"""
## Basic Usage
"""
logger.info("## Basic Usage")


# os.environ["OPENAI_API_KEY"] = "..."


llm = MLXResponses(
    model="qwen3-1.7b-4bit-mini",
)

"""
#### Call `complete` with a prompt
"""
logger.info("#### Call `complete` with a prompt")


resp = llm.complete("Paul Graham is ")

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
## Streaming

Using `stream_complete` endpoint
"""
logger.info("## Streaming")

resp = llm.stream_complete("Paul Graham is ")

for r in resp:
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
## Configure Parameters

The Respones API supports many options:
- Setting the model name
- Generation parameters like temperature, top_p, max_output_tokens
- enabling built-in tool calling
- setting the resoning effort for O-series models
- tracking previous responses for automatic conversation history
- and more!

### Basic Parameters
"""
logger.info("## Configure Parameters")


llm = MLXResponses(
    model="qwen3-1.7b-4bit-mini",
    temperature=0.5,  # default is 0.1
    max_output_tokens=100,  # default is None
    top_p=0.95,  # default is 1.0
)

"""
### Built-in Tool Calling

The responses API supports built-in tool calling, which you can read more about [here](https://platform.openai.com/docs/guides/tools?api-mode=responses).

Configuring this means that the LLM will automatically call the tool and use it to augment the response.

Tools are defined as a list of dictionaries, each containing settings for a tool.

Below is an example of using the built-in web search tool.
"""
logger.info("### Built-in Tool Calling")


llm = MLXResponses(
    model="qwen3-1.7b-4bit-mini",
    built_in_tools=[{"type": "web_search_preview"}],
)

resp = llm.chat(
    [ChatMessage(role="user", content="What is the weather in San Francisco?")]
)
logger.debug(resp)
logger.debug("========" * 2)
logger.debug(resp.additional_kwargs)

"""
## Reasoning Effort

For O-series models, you can set the reasoning effort to control the amount of time the model will spend reasoning.

See the [MLX API docs](https://platform.openai.com/docs/guides/reasoning?api-mode=responses) for more information.
"""
logger.info("## Reasoning Effort")


llm = MLXResponses(
    model="o3-mini",
    reasoning_options={"effort": "high"},
)

resp = llm.chat(
    [ChatMessage(role="user", content="What is the meaning of life?")]
)
logger.debug(resp)
logger.debug("========" * 2)
logger.debug(resp.additional_kwargs)

"""
## Image Support

MLX has support for images in the input of chat messages for many models.

Using the content blocks feature of chat messages, you can easily combone text and images in a single LLM prompt.
"""
logger.info("## Image Support")

# !wget https://cdn.pixabay.com/photo/2016/07/07/16/46/dice-1502706_640.jpg -O image.png


llm = MLXResponses(model="qwen3-1.7b-4bit")

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
## Using Function/Tool Calling

MLX models have native support for function calling. This conveniently integrates with LlamaIndex tool abstractions, letting you plug in any arbitrary Python function to the LLM.

In the example below, we define a function to generate a Song object.
"""
logger.info("## Using Function/Tool Calling")



class Song(BaseModel):
    """A song with name and artist"""

    name: str
    artist: str


def generate_song(name: str, artist: str) -> Song:
    """Generates a song with provided name and artist."""
    return Song(name=name, artist=artist)


tool = FunctionTool.from_defaults(fn=generate_song)

"""
The `strict` parameter tells MLX whether or not to use constrained sampling when generating tool calls/structured outputs. This means that the generated tool call schema will always contain the expected fields.

Since this seems to increase latency, it defaults to false.
"""
logger.info("The `strict` parameter tells MLX whether or not to use constrained sampling when generating tool calls/structured outputs. This means that the generated tool call schema will always contain the expected fields.")


llm = MLXResponses(model="qwen3-1.7b-4bit-mini", strict=True)
response = llm.predict_and_call(
    [tool],
    "Write a random song for me",
)
logger.debug(str(response))

"""
We can also do multiple function calling.
"""
logger.info("We can also do multiple function calling.")

llm = MLXResponses(model="qwen3-1.7b-4bit-mini")
response = llm.predict_and_call(
    [tool],
    "Generate five songs from the Beatles",
    allow_parallel_tool_calls=True,
)
for s in response.sources:
    logger.debug(f"Name: {s.tool_name}, Input: {s.raw_input}, Output: {str(s)}")

"""
### Manual Tool Calling

If you want to control how a tool is called, you can also split the tool calling and tool selection into their own steps.

First, lets select a tool.
"""
logger.info("### Manual Tool Calling")


llm = MLXResponses(model="qwen3-1.7b-4bit-mini")

chat_history = [ChatMessage(role="user", content="Write a random song for me")]

resp = llm.chat_with_tools([tool], chat_history=chat_history)

"""
Now, lets call the tool the LLM selected (if any).

If there was a tool call, we should send the results to the LLM to generate the final response (or another tool call!).
"""
logger.info("Now, lets call the tool the LLM selected (if any).")

tools_by_name = {t.metadata.name: t for t in [tool]}
tool_calls = llm.get_tool_calls_from_response(
    resp, error_on_no_tool_call=False
)

while tool_calls:
    chat_history.append(resp.message)

    for tool_call in tool_calls:
        tool_name = tool_call.tool_name
        tool_kwargs = tool_call.tool_kwargs

        logger.debug(f"Calling {tool_name} with {tool_kwargs}")
        tool_output = tool(**tool_kwargs)
        chat_history.append(
            ChatMessage(
                role="tool",
                content=str(tool_output),
                additional_kwargs={"call_id": tool_call.tool_id},
            )
        )

        resp = llm.chat_with_tools([tool], chat_history=chat_history)
        tool_calls = llm.get_tool_calls_from_response(
            resp, error_on_no_tool_call=False
        )

"""
Now, we should have a final response!
"""
logger.info("Now, we should have a final response!")

logger.debug(resp.message.content)

"""
## Structured Prediction

An important use case for function calling is extracting structured objects. LlamaIndex provides an intuitive interface for converting any LLM into a structured LLM - simply define the target Pydantic class (can be nested), and given a prompt, we extract out the desired object.
"""
logger.info("## Structured Prediction")



class MenuItem(BaseModel):
    """A menu item in a restaurant."""

    course_name: str
    is_vegetarian: bool


class Restaurant(BaseModel):
    """A restaurant with name, city, and cuisine."""

    name: str
    city: str
    cuisine: str
    menu_items: List[MenuItem]


llm = MLXResponses(model="qwen3-1.7b-4bit-mini")
prompt_tmpl = PromptTemplate(
    "Generate a restaurant in a given city {city_name}"
)
restaurant_obj = (
    llm.as_structured_llm(Restaurant)
    .complete(prompt_tmpl.format(city_name="Dallas"))
    .raw
)

restaurant_obj

"""
## Async
"""
logger.info("## Async")


llm = MLXResponses(model="qwen3-1.7b-4bit")

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

async def run_async_code_240f4fad():
    async def run_async_code_506ce1e2():
        resp = llm.stream_complete("Paul Graham is ")
        return resp
    resp = asyncio.run(run_async_code_506ce1e2())
    logger.success(format_json(resp))
    return resp
resp = asyncio.run(run_async_code_240f4fad())
logger.success(format_json(resp))

async for delta in resp:
    logger.debug(delta.delta, end="")

"""
Async function calling is also supported.
"""
logger.info("Async function calling is also supported.")

llm = MLXResponses(model="qwen3-1.7b-4bit-mini")
async def run_async_code_7be6247b():
    async def run_async_code_89fea8b9():
        response = llm.predict_and_call([tool], "Generate a random song")
        return response
    response = asyncio.run(run_async_code_89fea8b9())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_7be6247b())
logger.success(format_json(response))
logger.debug(str(response))

"""
## Additional kwargs

If there are additional kwargs not present in the constructor, you can set them at a per-instance level with `additional_kwargs`.

These will be passed into every call to the LLM.
"""
logger.info("## Additional kwargs")


llm = MLXResponses(
    model="qwen3-1.7b-4bit-mini", additional_kwargs={"user": "your_user_id"}
)
resp = llm.complete("Paul Graham is ")
logger.debug(resp)

"""
## Image generation

You can use [image generation](https://platform.openai.com/docs/guides/image-generation?image-generation-model=gpt-image-1#generate-images) by passing, as a built-in-tool, `{'type': 'image_generation'}` or, if you want to enable streaming, `{'type': 'image_generation', 'partial_images': 2}`:
"""
logger.info("## Image generation")


llm = MLXResponses(
    model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats", built_in_tools=[{"type": "image_generation"}]
)
messages = [
    ChatMessage.from_str(
        content="A llama dancing with a cat in a meadow", role="user"
    )
]
response = llm.chat(
    messages
async def run_async_code_48b7cc44():
    async def run_async_code_800f7ac5():
        )  # response = llm.chat(messages) for an implementation
        return )  # response
    )  # response = asyncio.run(run_async_code_800f7ac5())
    logger.success(format_json()  # response))
    return )  # response
)  # response = asyncio.run(run_async_code_48b7cc44())
logger.success(format_json()  # response))
for block in response.message.blocks:
    if isinstance(block, ImageBlock):
        with open("llama_and_cat_dancing.png", "wb") as f:
            f.write(bas64.b64decode(block.image))
    elif isinstance(block, TextBlock):
        logger.debug(block.text)

llm_stream = MLXResponses(
    model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats",
    built_in_tools=[{"type": "image_generation", "partial_images": 2}],
)
response = llm_stream.stream_chat(
    messages
async def run_async_code_5fcc0ce2():
    )  # response = await llm_stream.asteam_chat(messages) for an async implementation
    return )  # response
)  # response = asyncio.run(run_async_code_5fcc0ce2())
logger.success(format_json()  # response))
for event in response:
    for block in event.message.blocks:
        if isinstance(block, ImageBlock):
            with open(f"llama_and_cat_dancing_{block.detail}.png", "wb") as f:
                f.write(bas64.b64decode(block.image))
        elif isinstance(block, TextBlock):
            logger.debug(block.text)

"""
## MCP Remote calls

You can call any [remote MCP](https://platform.openai.com/docs/guides/tools-remote-mcp) through the MLX Responses API just by passing the MCP specifics as a built-in tool to the LLM
"""
logger.info("## MCP Remote calls")


llm = MLXResponses(
    model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats",
    built_in_tools=[
        {
            "type": "mcp",
            "server_label": "deepwiki",
            "server_url": "https://mcp.deepwiki.com/mcp",
            "require_approval": "never",
        }
    ],
)
messages = [
    ChatMessage.from_str(
        content="What transport protocols are supported in the 2025-03-26 version of the MCP spec?",
        role="user",
    )
]
response = llm.chat(messages)
logger.debug(response.message.content)
logger.debug(response.raw.output[0])

"""
## Code interpreter

You can use the [Code Interpreter](https://platform.openai.com/docs/guides/tools-code-interpreter) just by setting, as a built-in tool, `"type": "code_interpreter", "container": { "type": "auto" }`.
"""
logger.info("## Code interpreter")


llm = MLXResponses(
    model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats",
    built_in_tools=[
        {
            "type": "code_interpreter",
            "container": {"type": "auto"},
        }
    ],
)
messages = messages = [
    ChatMessage.from_str(
        content="I need to solve the equation 3x + 11 = 14. Can you help me?",
        role="user",
    )
]
response = llm.chat(messages)
logger.debug(response.message.content)
logger.debug(response.raw.output[0])

logger.info("\n\n[DONE]", bright=True)