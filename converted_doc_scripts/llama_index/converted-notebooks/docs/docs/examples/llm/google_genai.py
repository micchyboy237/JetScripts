import asyncio
from jet.transformers.formatters import format_json
from IPython.display import clear_output
from IPython.display import display
from PIL import Image
from datetime import datetime
from google import genai
from google.genai import types
from google.genai.types import CreateCachedContentConfig, Content, Part
from jet.logger import CustomLogger
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.llms import ChatMessage
from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
from llama_index.core.llms import DocumentBlock
from llama_index.core.prompts import PromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.llms.google_genai import GoogleGenAI
from pprint import pprint
from typing import List
import google.genai.types as types
import os
import shutil
import time


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/gemini.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Google GenAI

In this notebook, we show how to use the `google-genai` Python SDK with LlamaIndex to interact with Google GenAI models.

If you're opening this Notebook on colab, you will need to install LlamaIndex 🦙 and the `google-genai` Python SDK.
"""
logger.info("# Google GenAI")

# %pip install llama-index-llms-google-genai llama-index

"""
## Basic Usage

You will need to get an API key from [Google AI Studio](https://makersuite.google.com/app/apikey). Once you have one, you can either pass it explicity to the model, or use the `GOOGLE_API_KEY` environment variable.
"""
logger.info("## Basic Usage")


os.environ["GOOGLE_API_KEY"] = "..."

"""
## Basic Usage

You can call `complete` with a prompt:
"""
logger.info("## Basic Usage")


llm = GoogleGenAI(
    model="gemini-2.0-flash",
)

resp = llm.complete("Who is Paul Graham?")
logger.debug(resp)

"""
You can also call `chat` with a list of chat messages:
"""
logger.info("You can also call `chat` with a list of chat messages:")


messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="Tell me a story"),
]
llm = GoogleGenAI(model="gemini-2.0-flash")
resp = llm.chat(messages)

logger.debug(resp)

"""
## Streaming Support

Every method supports streaming through the `stream_` prefix.
"""
logger.info("## Streaming Support")


llm = GoogleGenAI(model="gemini-2.0-flash")

resp = llm.stream_complete("Who is Paul Graham?")
for r in resp:
    logger.debug(r.delta, end="")


messages = [
    ChatMessage(role="user", content="Who is Paul Graham?"),
]

resp = llm.stream_chat(messages)
for r in resp:
    logger.debug(r.delta, end="")

"""
## Async Usage

Every synchronous method has an async counterpart.
"""
logger.info("## Async Usage")


llm = GoogleGenAI(model="gemini-2.0-flash")

async def run_async_code_8ed469d1():
    async def run_async_code_47c41223():
        resp = llm.stream_complete("Who is Paul Graham?")
        return resp
    resp = asyncio.run(run_async_code_47c41223())
    logger.success(format_json(resp))
    return resp
resp = asyncio.run(run_async_code_8ed469d1())
logger.success(format_json(resp))
async for r in resp:
    logger.debug(r.delta, end="")

messages = [
    ChatMessage(role="user", content="Who is Paul Graham?"),
]

async def run_async_code_836a7d61():
    async def run_async_code_fd99c2e7():
        resp = llm.chat(messages)
        return resp
    resp = asyncio.run(run_async_code_fd99c2e7())
    logger.success(format_json(resp))
    return resp
resp = asyncio.run(run_async_code_836a7d61())
logger.success(format_json(resp))
logger.debug(resp)

"""
## Vertex AI Support

By providing the `region` and `project_id` parameters (either through environment variables or directly), you can enable usage through Vertex AI.
"""
logger.info("## Vertex AI Support")

# !export GOOGLE_GENAI_USE_VERTEXAI=true
# !export GOOGLE_CLOUD_PROJECT='your-project-id'
# !export GOOGLE_CLOUD_LOCATION='us-central1'


llm = GoogleGenAI(
    model="gemini-2.0-flash",
    vertexai_config={"project": "your-project-id", "location": "us-central1"},
    context_window=200000,
    max_tokens=512,
)

"""
## Cached Content Support

Google GenAI supports cached content for improved performance and cost efficiency when reusing large contexts across multiple requests. This is particularly useful for RAG applications, document analysis, and multi-turn conversations with consistent context.

#### Benefits

- **Faster responses**
- **Cost savings** through reduced input token usage
- **Consistent context** across multiple queries
- **Perfect for document analysis** with large files

#### Creating Cached Content

First, create cached content using the Google GenAI SDK:
"""
logger.info("## Cached Content Support")


client = genai.Client(api_key="your-api-key")

"""
Option 1: Upload Local Files
"""
logger.info("Option 1: Upload Local Files")

pdf_file = client.files.upload(file="./your_document.pdf")
while pdf_file.state.name == "PROCESSING":
    logger.debug("Waiting for PDF to be processed.")
    time.sleep(2)
    pdf_file = client.files.get(name=pdf_file.name)

cache = client.caches.create(
    model="gemini-2.0-flash-001",
    config=CreateCachedContentConfig(
        display_name="Document Analysis Cache",
        system_instruction=(
            "You are an expert document analyzer. Answer questions "
            "based on the provided documents with accuracy and detail."
        ),
        contents=[pdf_file],  # Direct file reference
        ttl="3600s",  # Cache for 1 hour
    ),
)

"""
Option 2: Multiple Files with Content Structure
"""
logger.info("Option 2: Multiple Files with Content Structure")

contents = [
    Content(
        role="user",
        parts=[
            Part.from_uri(
                file_uri="gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf",
                mime_type="application/pdf",
            ),
            Part.from_uri(
                file_uri="gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf",
                mime_type="application/pdf",
            ),
        ],
    )
]

cache = client.caches.create(
    model="gemini-2.0-flash-001",
    config=CreateCachedContentConfig(
        display_name="Multi-Document Cache",
        system_instruction=(
            "You are an expert researcher. Analyze and compare "
            "information across the provided documents."
        ),
        contents=contents,
        ttl="3600s",
    ),
)

logger.debug(f"Cache created: {cache.name}")
logger.debug(f"Cached tokens: {cache.usage_metadata.total_token_count}")

"""
Using Cached Content with LlamaIndex

Once you have created the cache, use it with LlamaIndex:
"""
logger.info("Using Cached Content with LlamaIndex")


llm = GoogleGenAI(
    model="gemini-2.0-flash-001",
    api_key="your-api-key",
    cached_content=cache.name,
)


message = ChatMessage(
    role="user", content="Summarize the key findings from Chapter 4."
)
response = llm.chat([message])
logger.debug(response)

"""
Using Cached Content in Generation Config

For request-level caching control:
"""
logger.info("Using Cached Content in Generation Config")


config = types.GenerateContentConfig(
    cached_content=cache.name, temperature=0.1, max_output_tokens=1024
)

llm = GoogleGenAI(model="gemini-2.0-flash-001", generation_config=config)

response = llm.complete("List the first five chapters of the document")
logger.debug(response)

"""
Cache Management
"""
logger.info("Cache Management")

caches = client.caches.list()
for cache_item in caches:
    logger.debug(f"Cache: {cache_item.display_name} ({cache_item.name})")
    logger.debug(f"Tokens: {cache_item.usage_metadata.total_token_count}")

cache_info = client.caches.get(name=cache.name)
logger.debug(f"Created: {cache_info.create_time}")
logger.debug(f"Expires: {cache_info.expire_time}")

client.caches.delete(name=cache.name)
logger.debug("Cache deleted")

"""
## Multi-Modal Support

Using `ChatMessage` objects, you can pass in images and text to the LLM.
"""
logger.info("## Multi-Modal Support")

# !wget https://cdn.pixabay.com/photo/2021/12/12/20/00/play-6865967_640.jpg -O image.jpg


llm = GoogleGenAI(model="gemini-2.0-flash")

messages = [
    ChatMessage(
        role="user",
        blocks=[
            ImageBlock(path="image.jpg"),
            TextBlock(text="What is in this image?"),
        ],
    )
]

resp = llm.chat(messages)
logger.debug(resp)

"""
You can also pass in documents.
"""
logger.info("You can also pass in documents.")


messages = [
    ChatMessage(
        role="user",
        blocks=[
            DocumentBlock(
                path="/path/to/your/test.pdf", mime_type="application/pdf"
            ),
            TextBlock(text="Describe the document in a sentence."),
        ],
    )
]

resp = llm.chat(messages)
logger.debug(resp)

"""
## Structured Prediction

LlamaIndex provides an intuitive interface for converting any LLM into a structured LLM through `structured_predict` - simply define the target Pydantic class (can be nested), and given a prompt, we extract out the desired object.
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


llm = GoogleGenAI(model="gemini-2.0-flash")
prompt_tmpl = PromptTemplate(
    "Generate a restaurant in a given city {city_name}"
)

restaurant_obj = (
    llm.as_structured_llm(Restaurant)
    .complete(prompt_tmpl.format(city_name="Miami"))
    .raw
)

logger.debug(restaurant_obj)

"""
#### Structured Prediction with Streaming

Any LLM wrapped with `as_structured_llm` supports streaming through `stream_chat`.
"""
logger.info("#### Structured Prediction with Streaming")


input_msg = ChatMessage.from_str("Generate a restaurant in San Francisco")

sllm = llm.as_structured_llm(Restaurant)
stream_output = sllm.stream_chat([input_msg])
for partial_output in stream_output:
    clear_output(wait=True)
    plogger.debug(partial_output.raw.dict())
    restaurant_obj = partial_output.raw

restaurant_obj

"""
## Tool/Function Calling

Google GenAI supports direct tool/function calling through the API. Using LlamaIndex, we can implement some core agentic tool calling patterns.
"""
logger.info("## Tool/Function Calling")


llm = GoogleGenAI(model="gemini-2.0-flash")


def get_current_time(timezone: str) -> dict:
    """Get the current time"""
    return {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": timezone,
    }


tool = FunctionTool.from_defaults(fn=get_current_time)

"""
We can simply do a single pass to call the tool and get the result:
"""
logger.info("We can simply do a single pass to call the tool and get the result:")

resp = llm.predict_and_call([tool], "What is the current time in New York?")
logger.debug(resp)

"""
We can also use lower-level APIs to implement an agentic tool-calling loop!
"""
logger.info("We can also use lower-level APIs to implement an agentic tool-calling loop!")

chat_history = [
    ChatMessage(role="user", content="What is the current time in New York?")
]
tools_by_name = {t.metadata.name: t for t in [tool]}

resp = llm.chat_with_tools([tool], chat_history=chat_history)
tool_calls = llm.get_tool_calls_from_response(
    resp, error_on_no_tool_call=False
)

if not tool_calls:
    logger.debug(resp)
else:
    while tool_calls:
        chat_history.append(resp.message)

        for tool_call in tool_calls:
            tool_name = tool_call.tool_name
            tool_kwargs = tool_call.tool_kwargs

            logger.debug(f"Calling {tool_name} with {tool_kwargs}")
            tool_output = tool.call(**tool_kwargs)
            logger.debug("Tool output: ", tool_output)
            chat_history.append(
                ChatMessage(
                    role="tool",
                    content=str(tool_output),
                    additional_kwargs={"tool_call_id": tool_call.tool_id},
                )
            )

            resp = llm.chat_with_tools([tool], chat_history=chat_history)
            tool_calls = llm.get_tool_calls_from_response(
                resp, error_on_no_tool_call=False
            )
    logger.debug("Final response: ", resp.message.content)

"""
We can also call multiple tools simultaneously in a single request, making it efficient for complex queries that require different types of information.
"""
logger.info("We can also call multiple tools simultaneously in a single request, making it efficient for complex queries that require different types of information.")

def get_temperature(city: str) -> dict:
    """Get the current temperature for a city"""
    return {
        "city": city,
        "temperature": "25°C",
    }


tool1 = FunctionTool.from_defaults(fn=get_current_time)
tool2 = FunctionTool.from_defaults(fn=get_temperature)

chat_history = [
    ChatMessage(
        role="user",
        content="What is the current time and temperature in New York?",
    )
]

resp = llm.chat_with_tools([tool1, tool2], chat_history=chat_history)
tool_calls = llm.get_tool_calls_from_response(
    resp, error_on_no_tool_call=False
)

logger.debug(f"Model made {len(tool_calls)} tool calls:")
for i, tool_call in enumerate(tool_calls, 1):
    logger.debug(f"{i}. {tool_call.tool_name} with args: {tool_call.tool_kwargs}")

"""
## Google Search Grounding

Google Gemini 2.0 and 2.5 models support Google Search grounding, which allows the model to search for real-time information and ground its responses with web search results. This is particularly useful for getting up-to-date information.

The `built_in_tool` parameter accepts Google Search tools that enable the model to ground its responses with real-world data from Google Search results.
"""
logger.info("## Google Search Grounding")


grounding_tool = types.Tool(google_search=types.GoogleSearch())

llm = GoogleGenAI(
    model="gemini-2.0-flash",
    built_in_tool=grounding_tool,
)

resp = llm.complete("When is the next total solar eclipse in the US?")
logger.debug(resp)

"""
The Google Search grounding tool provides several benefits:

- **Real-time information**: Access to current events and up-to-date data
- **Factual accuracy**: Responses grounded in actual search results
- **Source attribution**: Grounding metadata includes search sources
- **Automatic search decisions**: The model determines when to search based on the query

You can also use the grounding tool with chat messages:
"""
logger.info("The Google Search grounding tool provides several benefits:")

messages = [ChatMessage(role="user", content="Who won the Euro 2024?")]

resp = llm.chat(messages)
logger.debug(resp)

if hasattr(resp, "raw") and "grounding_metadata" in resp.raw:
    logger.debug(resp.raw["grounding_metadata"])
else:
    logger.debug("\nNo grounding metadata in this response")

"""
## Code Execution

The `built_in_tool` parameter also accepts code execution tools that enable the model to write and execute Python code to solve problems, perform calculations, and analyze data. This is particularly useful for mathematical computations, data analysis, and generating visualizations.
"""
logger.info("## Code Execution")


code_execution_tool = types.Tool(code_execution=types.ToolCodeExecution())

llm = GoogleGenAI(
    model="gemini-2.0-flash",
    built_in_tool=code_execution_tool,
)

resp = llm.complete("Calculate 20th fibonacci number.")
logger.debug(resp)

"""
### Accessing Code Execution Details

When the model uses code execution, you can access the executed code, results, and other metadata through the raw response. This includes:

- **executable_code**: The actual Python code that was executed
- **code_execution_result**: The output from running the code
- **text**: The model's explanation and commentary

Let's see this in action:
"""
logger.info("### Accessing Code Execution Details")

messages = [
    ChatMessage(
        role="user", content="What is the sum of the first 50 prime numbers?"
    )
]

resp = llm.chat(messages)

if hasattr(resp, "raw") and "content" in resp.raw:
    parts = resp.raw["content"].get("parts", [])

    for i, part in enumerate(parts):
        logger.debug(f"Part {i+1}:")

        if "text" in part and part["text"]:
            logger.debug(f"  Text: {part['text'][:100]}", end="")
            logger.debug(" ..." if len(part["text"]) > 100 else "")

        if "executable_code" in part and part["executable_code"]:
            logger.debug(f"  Executable Code: {part['executable_code']}")

        if "code_execution_result" in part and part["code_execution_result"]:
            logger.debug(f"  Code Result: {part['code_execution_result']}")
else:
    logger.debug("No detailed parts found in raw response")

"""
## Image Generation

Select models also support image outputs, as well as image inputs. Using the `response_modalities` config, we can generate and edit images with a Gemini model!
"""
logger.info("## Image Generation")


config = types.GenerateContentConfig(
    temperature=0.1, response_modalities=["Text", "Image"]
)

llm = GoogleGenAI(
    model="models/gemini-2.0-flash-exp", generation_config=config
)


messages = [
    ChatMessage(role="user", content="Please generate an image of a cute dog")
]

resp = llm.chat(messages)


for block in resp.message.blocks:
    if isinstance(block, ImageBlock):
        image = Image.open(block.resolve_image())
        display(image)
    elif isinstance(block, TextBlock):
        logger.debug(block.text)

"""
We can also edit the image!
"""
logger.info("We can also edit the image!")

messages.append(resp.message)
messages.append(
    ChatMessage(
        role="user",
        content="Please edit the image to make the dog a mini-schnauzer, but keep the same overall pose, framing, background, and art style.",
    )
)

resp = llm.chat(messages)

for block in resp.message.blocks:
    if isinstance(block, ImageBlock):
        image = Image.open(block.resolve_image())
        display(image)
    elif isinstance(block, TextBlock):
        logger.debug(block.text)

logger.info("\n\n[DONE]", bright=True)