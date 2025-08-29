from jet.models.config import MODELS_CACHE_DIR
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms import ChatMessage
from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
from llama_index.core.prompts import PromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from llama_index.llms.asi import ASI
from pydantic import BaseModel
from typing import List
import asyncio
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/asi1.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ASI LLM
ASI1-Mini is an advanced, agentic LLM designed by fetch.ai, a founding member of Artificial Superintelligence Alliance for decentralized operations. Its unique architecture empowers it to execute tasks and collaborate with other agents for efficient, adaptable problem-solving in complex environments.

This notebook demonstrates how to use ASI models with LlamaIndex. It covers various functionalities including basic completion, chat, streaming, function calling, structured prediction, RAG, and more.
If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

## Setup

First, let's install the required packages:
"""
logger.info("# ASI LLM")

# %pip install llama-index-llms-asi llama-index-llms-ollama llama-index-core

"""
## Setting API Keys

You'll need to set your API keys for ASI and optionally for OllamaFunctionCallingAdapter if you want to compare the two:
"""
logger.info("## Setting API Keys")


os.environ["ASI_API_KEY"] = "your-api-key"

"""
## Basic Completion

Let's start with a basic completion example using ASI:
"""
logger.info("## Basic Completion")


llm = ASI(model="asi1-mini")

response = llm.complete("Who is Paul Graham? ")
logger.debug(response)

"""
## Chat

Now let's try chat functionality:
"""
logger.info("## Chat")


messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]

chat_response = llm.chat(messages)
logger.debug(chat_response)

"""
## Streaming

ASI supports streaming for chat responses:
"""
logger.info("## Streaming")

for chunk in llm.stream_chat(messages):
    logger.debug(chunk.delta, end="")

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
Using `stream_complete` endpoint
"""
logger.info("Using `stream_complete` endpoint")

resp = llm.stream_complete("Paul Graham is ")

for r in resp:
    logger.debug(r.delta, end="")

"""
## Image Support

ASI has support for images in the input of chat messages for many models.

Using the content blocks feature of chat messages, you can easily combone text and images in a single LLM prompt.
"""
logger.info("## Image Support")

# !wget https://cdn.pixabay.com/photo/2016/07/07/16/46/dice-1502706_640.jpg -O image.png


llm = ASI(model="asi1-mini")

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
## Function Calling/Tool Calling

ASI LLM have native support for function calling. This conveniently integrates with LlamaIndex tool abstractions, letting you plug in any arbitrary Python function to the LLM.

In the example below, we define a function to generate a Song object.
"""
logger.info("## Function Calling/Tool Calling")



class Song(BaseModel):
    """A song with name and artist"""

    name: str
    artist: str


def generate_song(name: str, artist: str) -> Song:
    """Generates a song with provided name and artist."""
    return Song(name="Sky full of stars", artist="Coldplay")


tool = FunctionTool.from_defaults(fn=generate_song)

"""
The strict parameter tells ASI whether or not to use constrained sampling when generating tool calls/structured outputs. This means that the generated tool call schema will always contain the expected fields.

Since this seems to increase latency, it defaults to false.
"""
logger.info("The strict parameter tells ASI whether or not to use constrained sampling when generating tool calls/structured outputs. This means that the generated tool call schema will always contain the expected fields.")


llm = ASI(model="asi1-mini", strict=True)
response = llm.predict_and_call(
    [tool],
    "Pick a random song for me",
)
logger.debug(str(response))

llm = ASI(model="asi1-mini")
response = llm.predict_and_call(
    [tool],
    "Generate five songs from the Beatles",
    allow_parallel_tool_calls=True,
)
for s in response.sources:
    logger.debug(f"Name: {s.tool_name}, Input: {s.raw_input}, Output: {str(s)}")

"""
## Manual Tool Calling

While automatic tool calling with `predict_and_call` provides a streamlined experience, manual tool calling gives you more control over the process. With manual tool calling, you can:

1. Explicitly control when and how tools are called
2. Process intermediate results before continuing the conversation
3. Implement custom error handling and fallback strategies
4. Chain multiple tool calls together in a specific sequence

ASI supports manual tool calling, but requires more specific prompting compared to some other LLMs. For best results with ASI, include a system message that explains the available tools and provide specific parameters in your user prompt.

The following example demonstrates manual tool calling with ASI to generate a song:
"""
logger.info("## Manual Tool Calling")



class Song(BaseModel):
    """A song with name and artist"""

    name: str
    artist: str


def generate_song(name: str, artist: str) -> Song:
    """Generates a song with provided name and artist."""
    return Song(name=name, artist=artist)


tool = FunctionTool.from_defaults(fn=generate_song)

chat_history = [
    ChatMessage(
        role="system",
        content="You have access to a tool called generate_song that can create songs. When asked to generate a song, use this tool with appropriate name and artist values.",
    ),
    ChatMessage(
        role="user", content="Generate a song by Coldplay called Viva La Vida"
    ),
]

resp = llm.chat_with_tools([tool], chat_history=chat_history)
logger.debug(f"Initial response: {resp.message.content}")

tool_calls = llm.get_tool_calls_from_response(
    resp, error_on_no_tool_call=False
)

if tool_calls:
    chat_history.append(resp.message)

    for tool_call in tool_calls:
        tool_name = tool_call.tool_name
        tool_kwargs = tool_call.tool_kwargs

        logger.debug(f"Calling {tool_name} with {tool_kwargs}")
        tool_output = tool(**tool_kwargs)
        logger.debug(f"Tool output: {tool_output}")

        chat_history.append(
            ChatMessage(
                role="tool",
                content=str(tool_output),
                additional_kwargs={"tool_call_id": tool_call.tool_id},
            )
        )

        resp = llm.chat_with_tools([tool], chat_history=chat_history)
        logger.debug(f"Final response: {resp.message.content}")
else:
    logger.debug("No tool calls detected in the response.")

"""
## Structured Prediction

You can use ASI to extract structured data from text:
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


prompt_tmpl = PromptTemplate(
    "Generate a restaurant in a given city {city_name}"
)

restaurant_obj = llm.structured_predict(
    Restaurant, prompt_tmpl, city_name="Dallas"
)
logger.debug(f"Restaurant: {restaurant_obj}")

structured_llm = llm.as_structured_llm(Restaurant)
restaurant_obj2 = structured_llm.complete(
    prompt_tmpl.format(city_name="Miami")
).raw
logger.debug(f"Restaurant: {restaurant_obj2}")

"""
**Note:** Structured streaming is currently not supported with ASI.

## Async

ASI supports async operations:
"""
logger.info("## Async")


llm = ASI(model="asi1-mini")

resp = llm.complete("who is Paul Graham")
logger.success(format_json(resp))

logger.debug(resp)

resp = llm.stream_complete("Paul Graham is ")
logger.success(format_json(resp))

# import nest_asyncio

async for delta in resp:
    logger.debug(delta.delta, end="")

# import nest_asyncio

# nest_asyncio.apply()


async def test_async():
    resp = llm.complete("Paul Graham is ")
    logger.success(format_json(resp))
    logger.debug(f"Async completion: {resp}")

    resp = llm.chat(messages)
    logger.success(format_json(resp))
    logger.debug(f"Async chat: {resp}")

    logger.debug("Async streaming completion: ", end="")
    resp = llm.stream_complete("Paul Graham is ")
    logger.success(format_json(resp))
    async for delta in resp:
        logger.debug(delta.delta, end="")
    logger.debug()

    logger.debug("Async streaming chat: ", end="")
    resp = llm.stream_chat(messages)
    logger.success(format_json(resp))
    async for delta in resp:
        logger.debug(delta.delta, end="")
    logger.debug()


asyncio.run(test_async())

"""
## Simple RAG

Let's implement a simple RAG application with ASI:
"""
logger.info("## Simple RAG")

# %pip install llama-index-embeddings-huggingface


# os.environ["OPENAI_API_KEY"] = "your-api-key"
# !mkdir -p temp_data
# !echo "Paul Graham is a programmer, writer, and investor. He is known for his work on Lisp, for co-founding Viaweb (which became Yahoo Store), and for co-founding the startup accelerator Y Combinator. He is also known for his essays on his website. He studied at HolaHola High school" > temp_data/paul_graham.txt

documents = SimpleDirectoryReader("temp_data").load_data()

llm = ASI(model="asi1-mini")
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR),  # Using OllamaFunctionCallingAdapter for embeddings
    llm=llm,  # Using ASI for generation
)

query_engine = index.as_query_engine()

response = query_engine.query("Where did Paul Graham study?")
logger.debug(response)

"""
## LlamaCloud RAG

If you have a LlamaCloud account, you can use ASI with LlamaCloud for RAG:
"""
logger.info("## LlamaCloud RAG")

# %pip install llama-cloud llama-index-indices-managed-llama-cloud


os.environ["LLAMA_CLOUD_API_KEY"] = "your-key"
# os.environ["OPENAI_API_KEY"] = "your-key"



try:
    index = LlamaCloudIndex(
        name="your-index-naem",
        project_name="Default",
        organization_id="your-id",
        api_key=os.environ["LLAMA_CLOUD_API_KEY"],
    )
    logger.debug("Successfully connected to LlamaCloud index")

    llm = ASI(model="asi1-mini")

    retriever = index.as_retriever()

    query_engine = index.as_query_engine(llm=llm)

    query = "What is the revenue of Uber in 2021?"
    logger.debug(f"\nTesting retriever with query: {query}")
    nodes = retriever.retrieve(query)
    logger.debug(f"Retrieved {len(nodes)} nodes\n")

    for i, node in enumerate(nodes[:3]):
        logger.debug(f"Node {i+1}:")
        logger.debug(f"Node ID: {node.node_id}")
        logger.debug(f"Score: {node.score}")
        logger.debug(f"Text: {node.text[:200]}...\n")

    logger.debug(f"Testing query engine with query: {query}")
    response = query_engine.query(query)
    logger.debug(f"Response: {response}")
except Exception as e:
    logger.debug(f"Error: {e}")

"""
## Set API Key at a per-instance level

If desired, you can have separate LLM instances use separate API keys:
"""
logger.info("## Set API Key at a per-instance level")


llm = ASI(model="asi1-mini", api_key="your_specific_api_key")

try:
    resp = llm.complete("Paul Graham is ")
    logger.debug(resp)
except Exception as e:
    logger.debug(f"Error with invalid API key: {e}")

"""
## Additional kwargs

Rather than adding the same parameters to each chat or completion call, you can set them at a per-instance level with additional_kwargs:
"""
logger.info("## Additional kwargs")


llm = ASI(model="asi1-mini", additional_kwargs={"user": "your_user_id"})

resp = llm.complete("Paul Graham is ")
logger.debug(resp)


llm = ASI(model="asi1-mini", additional_kwargs={"user": "your_user_id"})

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]

resp = llm.chat(messages)
logger.debug(resp)

"""
## Conclusion

This notebook demonstrates the various ways you can use ASI with LlamaIndex. The integration supports most of the functionality available in LlamaIndex, including:

- Basic completion and chat
- Streaming responses
- Multimodal support
- Function calling
- Structured prediction
- Async operations
- RAG applications
- LlamaCloud integration
- Per-instance API keys
- Additional kwargs

Note that structured streaming is currently not supported with ASI.
"""
logger.info("## Conclusion")

logger.info("\n\n[DONE]", bright=True)