from jet.transformers.formatters import format_json
from IPython.display import Audio
from IPython.display import clear_output
from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
from jet.logger import CustomLogger
from llama_cloud.client import LlamaCloud
from llama_index.core.llms import ChatMessage
from llama_index.core.llms import ChatMessage, AudioBlock, TextBlock
from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
from llama_index.core.prompts import PromptTemplate
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.tools import FunctionTool
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from pprint import pprint
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

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/openai.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# OllamaFunctionCalling

This notebook shows how to use the OllamaFunctionCalling LLM.

If you are looking to integrate with an OllamaFunctionCalling-Compatible API that is not the official OllamaFunctionCalling API, please see the [OllamaFunctionCalling-Compatible LLMs](https://docs.llamaindex.ai/en/stable/api_reference/llms/openai_like/#jet.llm.ollama.adapters.ollama_llama_index_llm_adapter_like.OllamaFunctionCallingAdapterLike) integration.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# OllamaFunctionCalling")

# %pip install llama-index llama-index-llms-ollama

"""
## Basic Usage
"""
logger.info("## Basic Usage")


# os.environ["OPENAI_API_KEY"] = "sk-..."


llm = OllamaFunctionCalling(
    model="llama3.2",
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
## Configure Model
"""
logger.info("## Configure Model")


llm = OllamaFunctionCalling(model="llama3.2")

resp = llm.complete("Paul Graham is ")

logger.debug(resp)

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.chat(messages)

logger.debug(resp)

"""
## Image Support

OllamaFunctionCalling has support for images in the input of chat messages for many models.

Using the content blocks feature of chat messages, you can easily combone text and images in a single LLM prompt.
"""
logger.info("## Image Support")

# !wget https://cdn.pixabay.com/photo/2016/07/07/16/46/dice-1502706_640.jpg -O image.png


llm = OllamaFunctionCalling(model="llama3.2")

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
## Audio Support

OllamaFunctionCalling has beta support for audio inputs and outputs, using their audio-preview models.

When using these models, you can configure the output modality (text or audio) using the `modalities` parameter. The output audio configuration can also be set using the `audio_config` parameter. See the [OllamaFunctionCalling docs](https://platform.openai.com/docs/guides/audio) for more information.
"""
logger.info("## Audio Support")


llm = OllamaFunctionCalling(
    model="llama3.2", request_timeout=300.0, context_window=4096,
    modalities=["text", "audio"],
    audio_config={"voice": "alloy", "format": "wav"},
)

messages = [
    ChatMessage(role="user", content="Hello! My name is Logan."),
]

resp = llm.chat(messages)


Audio(base64.b64decode(resp.message.blocks[0].audio), rate=16000)

messages.append(resp.message)
messages.append(ChatMessage(role="user", content="What is my name?"))

resp = llm.chat(messages)
Audio(base64.b64decode(resp.message.blocks[0].audio), rate=16000)

"""
We can also use audio as input and get descriptions or transcriptions of the audio.
"""
logger.info(
    "We can also use audio as input and get descriptions or transcriptions of the audio.")

# !wget AUDIO_URL = "https://science.nasa.gov/wp-content/uploads/2024/04/sounds-of-mars-one-small-step-earth.wav" -O audio.wav


messages = [
    ChatMessage(
        role="user",
        blocks=[
            AudioBlock(path="audio.wav", format="wav"),
            TextBlock(
                text="Describe the audio in a few sentences. What is it from?"
            ),
        ],
    )
]

llm = OllamaFunctionCalling(
    model="llama3.2", request_timeout=300.0, context_window=4096,
    modalities=["text"],
)

resp = llm.chat(messages)
logger.debug(resp)

"""
## Using Function/Tool Calling

OllamaFunctionCalling models have native support for function calling. This conveniently integrates with LlamaIndex tool abstractions, letting you plug in any arbitrary Python function to the LLM.

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
The `strict` parameter tells OllamaFunctionCalling whether or not to use constrained sampling when generating tool calls/structured outputs. This means that the generated tool call schema will always contain the expected fields.

Since this seems to increase latency, it defaults to false.
"""
logger.info("The `strict` parameter tells OllamaFunctionCalling whether or not to use constrained sampling when generating tool calls/structured outputs. This means that the generated tool call schema will always contain the expected fields.")


llm = OllamaFunctionCalling(model="llama3.2", strict=True)
response = llm.predict_and_call(
    [tool],
    "Pick a random song for me",
)
logger.debug(str(response))

"""
We can also do multiple function calling.
"""
logger.info("We can also do multiple function calling.")

llm = OllamaFunctionCalling(model="llama3.2")
response = llm.predict_and_call(
    [tool],
    "Generate five songs from the Beatles",
    allow_parallel_tool_calls=True,
)
for s in response.sources:
    logger.debug(
        f"Name: {s.tool_name}, Input: {s.raw_input}, Output: {str(s)}")

"""
### Manual Tool Calling

If you want to control how a tool is called, you can also split the tool calling and tool selection into their own steps.

First, lets select a tool.
"""
logger.info("### Manual Tool Calling")


chat_history = [ChatMessage(role="user", content="Pick a random song for me")]

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
                additional_kwargs={"tool_call_id": tool_call.tool_id},
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


llm = OllamaFunctionCalling(model="llama3.2")
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
#### Structured Prediction with Streaming

Any LLM wrapped with `as_structured_llm` supports streaming through `stream_chat`.
"""
logger.info("#### Structured Prediction with Streaming")


input_msg = ChatMessage.from_str("Generate a restaurant in Boston")

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


llm = OllamaFunctionCalling(model="llama3.2")

resp = llm.complete("Paul Graham is ")
logger.success(format_json(resp))

logger.debug(resp)

resp = llm.stream_complete("Paul Graham is ")
logger.success(format_json(resp))

async for delta in resp:
    logger.debug(delta.delta, end="")

"""
Async function calling is also supported.
"""
logger.info("Async function calling is also supported.")

llm = OllamaFunctionCalling(model="llama3.2")
response = llm.predict_and_call([tool], "Generate a song")
logger.success(format_json(response))
logger.debug(str(response))

"""
## Set API Key at a per-instance level
If desired, you can have separate LLM instances use separate API keys.
"""
logger.info("## Set API Key at a per-instance level")


llm = OllamaFunctionCalling(
    model="llama3.2", request_timeout=300.0, context_window=4096, api_key="BAD_KEY")
resp = llm.complete("Paul Graham is ")
logger.debug(resp)

"""
## Additional kwargs
Rather than adding same parameters to each chat or completion call, you can set them at a per-instance level with `additional_kwargs`.
"""
logger.info("## Additional kwargs")


llm = OllamaFunctionCalling(model="llama3.2", request_timeout=300.0,
                                   context_window=4096, additional_kwargs={"user": "your_user_id"})
resp = llm.complete("Paul Graham is ")
logger.debug(resp)


llm = OllamaFunctionCalling(model="llama3.2", request_timeout=300.0,
                                   context_window=4096, additional_kwargs={"user": "your_user_id"})
messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]
resp = llm.chat(messages)

"""
## RAG with LlamaCloud

LlamaCloud is our cloud-based service that allows you to upload, parse, and index documents, and then search them using LlamaIndex.  LlamaCloud is currently in a private alpha; please [get in touch](https://docs.google.com/forms/d/e/1FAIpQLSdehUJJB4NIYfrPIKoFdF4j8kyfnLhMSH_qYJI_WGQbDWD25A/viewform) if you'd like to be considered as a design partner.

### Installation
"""
logger.info("## RAG with LlamaCloud")

# %pip install llama-cloud
# %pip install llama-index-indices-managed-llama-cloud

"""
### Setup OllamaFunctionCalling and LlamaCloud API Keys
"""
logger.info("### Setup OllamaFunctionCalling and LlamaCloud API Keys")


# os.environ["OPENAI_API_KEY"] = "sk-..."

os.environ["LLAMA_CLOUD_API_KEY"] = "llx-..."


client = LlamaCloud(token=os.environ["LLAMA_CLOUD_API_KEY"])

"""
### Create a Pipeline.

Pipeline is an empty index on which you can ingest data.


You need to Setup transformation and embedding config which will be used while ingesting the data.
"""
logger.info("### Create a Pipeline.")

embedding_config = {
    "type": "OPENAI_EMBEDDING",
    "component": {
        #         "api_key": os.environ["OPENAI_API_KEY"],
        # You can choose any OllamaFunctionCalling Embedding model
        "model_name": "text-embedding-ada-002",
    },
}

transform_config = {
    "mode": "auto",
    "config": {
        "chunk_size": 1024,  # editable
        "chunk_overlap": 20,  # editable
    },
}

pipeline = {
    "name": "openai-rag-pipeline",  # Change the name if needed
    "embedding_config": embedding_config,
    "transform_config": transform_config,
    "data_sink_id": None,
}

pipeline = client.pipelines.upsert_pipeline(request=pipeline)

"""
### File Upload

We will upload files and add them to the index.
"""
logger.info("### File Upload")

with open("../data/10k/uber_2021.pdf", "rb") as f:
    file = client.files.upload_file(upload_file=f)

files = [{"file_id": file.id}]

pipeline_files = client.pipelines.add_files_to_pipeline(
    pipeline.id, request=files
)

"""
### Check the Ingestion job status
"""
logger.info("### Check the Ingestion job status")

jobs = client.pipelines.list_pipeline_jobs(pipeline.id)

jobs[0].status

"""
### Connect to Index.

Once the ingestion job is done, head over to your index on the [platform](https://cloud.llamaindex.ai/) and get the necessary details to connect to the index.
"""
logger.info("### Connect to Index.")


index = LlamaCloudIndex(
    name="openai-rag-pipeline",
    project_name="Default",
    organization_id="YOUR ORG ID",
    api_key=os.environ["LLAMA_CLOUD_API_KEY"],
)

"""
### Test on Sample Query
"""
logger.info("### Test on Sample Query")

query = "What is the revenue of Uber in 2021?"

"""
### Retriever 

Here we use hybrid search and re-ranker (cohere re-ranker by default).
"""
logger.info("### Retriever")

retriever = index.as_retriever(
    dense_similarity_top_k=3,
    sparse_similarity_top_k=3,
    alpha=0.5,
    enable_reranking=True,
)

retrieved_nodes = retriever.retrieve(query)

"""
#### Display the retrieved nodes
"""
logger.info("#### Display the retrieved nodes")


for retrieved_node in retrieved_nodes:
    display_source_node(retrieved_node, source_length=1000)

"""
#### Query Engine

QueryEngine to setup entire RAG workflow.
"""
logger.info("#### Query Engine")

query_engine = index.as_query_engine(
    dense_similarity_top_k=3,
    sparse_similarity_top_k=3,
    alpha=0.5,
    enable_reranking=True,
)

"""
#### Response
"""
logger.info("#### Response")

response = query_engine.query(query)

logger.debug(response)

logger.info("\n\n[DONE]", bright=True)
