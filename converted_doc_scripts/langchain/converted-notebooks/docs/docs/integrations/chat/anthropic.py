from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama import convert_to_anthropic_tool
from jet.logger import logger
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import MarkdownTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, add_messages
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from typing import Literal
from typing_extensions import Annotated, TypedDict
import anthropic
import json
import os
import requests
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
sidebar_label: Ollama
---

# ChatOllama

This notebook provides a quick overview for getting started with Ollama [chat models](/docs/concepts/chat_models). For detailed documentation of all ChatOllama features and configurations head to the [API reference](https://python.langchain.com/api_reference/anthropic/chat_models/jet.adapters.langchain.chat_ollama.chat_models.ChatOllama.html).

Ollama has several chat models. You can find information about their latest models and their costs, context windows, and supported input types in the [Ollama docs](https://docs.anthropic.com/en/docs/models-overview).


:::info AWS Bedrock and Google VertexAI

Note that certain Ollama models can also be accessed via AWS Bedrock and Google VertexAI. See the [ChatBedrock](/docs/integrations/chat/bedrock/) and [ChatVertexAI](/docs/integrations/chat/google_vertex_ai_palm/) integrations to use Ollama models via these services.

:::

## Overview
### Integration details

| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/anthropic) | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [ChatOllama](https://python.langchain.com/api_reference/anthropic/chat_models/jet.adapters.langchain.chat_ollama.chat_models.ChatOllama.html) | [langchain-anthropic](https://python.langchain.com/api_reference/anthropic/index.html) | ❌ | beta | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-anthropic?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-anthropic?style=flat-square&label=%20) |

### Model features
| [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
| :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
| ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |

## Setup

To access Ollama models you'll need to create an Ollama account, get an API key, and install the `langchain-anthropic` integration package.

### Credentials

# Head to https://console.anthropic.com/ to sign up for Ollama and generate an API key. Once you've done this set the ANTHROPIC_API_KEY environment variable:
"""
logger.info("# ChatOllama")

# import getpass

# if "ANTHROPIC_API_KEY" not in os.environ:
#     os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter your Ollama API key: ")

"""
To enable automated tracing of your model calls, set your [LangSmith](https://docs.smith.langchain.com/) API key:
"""
logger.info(
    "To enable automated tracing of your model calls, set your [LangSmith](https://docs.smith.langchain.com/) API key:")


"""
### Installation

The LangChain Ollama integration lives in the `langchain-anthropic` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-anthropic

"""
:::info This guide requires ``langchain-anthropic>=0.3.13``

:::

## Instantiation

Now we can instantiate our model object and generate chat completions:
"""
logger.info("## Instantiation")


llm = ChatOllama(
    model="llama3.2",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
)

"""
## Invocation
"""
logger.info("## Invocation")

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
ai_msg

logger.debug(ai_msg.content)

"""
## Chaining

We can [chain](/docs/how_to/sequence/) our model with a prompt template like so:
"""
logger.info("## Chaining")


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | llm
chain.invoke(
    {
        "input_language": "English",
        "output_language": "German",
        "input": "I love programming.",
    }
)

"""
## Content blocks

Content from a single Ollama AI message can either be a single string or a **list of content blocks**. For example when an Ollama model invokes a tool, the tool invocation is part of the message content (as well as being exposed in the standardized `AIMessage.tool_calls`):
"""
logger.info("## Content blocks")


class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(...,
                          description="The city and state, e.g. San Francisco, CA")


llm_with_tools = llm.bind_tools([GetWeather])
ai_msg = llm_with_tools.invoke("Which city is hotter today: LA or NY?")
ai_msg.content

ai_msg.tool_calls

"""
## Multimodal

Claude supports image and PDF inputs as content blocks, both in Ollama's native format (see docs for [vision](https://docs.anthropic.com/en/docs/build-with-claude/vision#base64-encoded-image-example) and [PDF support](https://docs.anthropic.com/en/docs/build-with-claude/pdf-support)) as well as LangChain's [standard format](/docs/how_to/multimodal_inputs/).

### Files API

Claude also supports interactions with files through its managed [Files API](https://docs.anthropic.com/en/docs/build-with-claude/files). See examples below.

The Files API can also be used to upload files to a container for use with Claude's built-in code-execution tools. See the [code execution](#code-execution) section below, for details.

<details>
<summary>Images</summary>

```python
# Upload image


client = anthropic.Ollama()
file = client.beta.files.upload(
    # Supports image/jpeg, image/png, image/gif, image/webp
    file=("image.png", open("/path/to/image.png", "rb"), "image/png"),
)
image_file_id = file.id


# Run inference

llm = ChatOllama(
    model="claude-sonnet-4-20250514",
    betas=["files-api-2025-04-14"],
)

input_message = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "Describe this image.",
        },
        {
            "type": "image",
            "source": {
                "type": "file",
                "file_id": image_file_id,
            },
        },
    ],
}
llm.invoke([input_message])
```

</details>

<details>
<summary>PDFs</summary>

```python
# Upload document


client = anthropic.Ollama()
file = client.beta.files.upload(
    file=("document.pdf", open("/path/to/document.pdf", "rb"), "application/pdf"),
)
pdf_file_id = file.id


# Run inference

llm = ChatOllama(
    model="claude-sonnet-4-20250514",
    betas=["files-api-2025-04-14"],
)

input_message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe this document."},
        {"type": "document", "source": {"type": "file", "file_id": pdf_file_id}}
    ],
}
llm.invoke([input_message])
```

</details>

## Extended thinking

Claude 3.7 Sonnet supports an [extended thinking](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking) feature, which will output the step-by-step reasoning process that led to its final answer.

To use it, specify the `thinking` parameter when initializing `ChatOllama`. It can also be passed in as a kwarg during invocation.

You will need to specify a token budget to use this feature. See usage example below:
"""
logger.info("## Multimodal")


llm = ChatOllama(
    model="claude-3-7-sonnet-latest",
    max_tokens=5000,
    thinking={"type": "enabled", "budget_tokens": 2000},
)

response = llm.invoke("What is the cube root of 50.653?")
logger.debug(json.dumps(response.content, indent=2))

"""
## Prompt caching

Ollama supports [caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) of [elements of your prompts](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#what-can-be-cached), including messages, tool definitions, tool results, images and documents. This allows you to re-use large documents, instructions, [few-shot documents](/docs/concepts/few_shot_prompting/), and other data to reduce latency and costs.

To enable caching on an element of a prompt, mark its associated content block using the `cache_control` key. See examples below:

### Messages
"""
logger.info("## Prompt caching")


llm = ChatOllama(model="llama3.2")

get_response = requests.get(
    "https://raw.githubusercontent.com/langchain-ai/langchain/master/README.md"
)
readme = get_response.text

messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are a technology expert.",
            },
            {
                "type": "text",
                "text": f"{readme}",
                "cache_control": {"type": "ephemeral"},
            },
        ],
    },
    {
        "role": "user",
        "content": "What's LangChain, according to its README?",
    },
]

response_1 = llm.invoke(messages)
response_2 = llm.invoke(messages)

usage_1 = response_1.usage_metadata["input_token_details"]
usage_2 = response_2.usage_metadata["input_token_details"]

logger.debug(f"First invocation:\n{usage_1}")
logger.debug(f"\nSecond:\n{usage_2}")

"""
:::tip Extended caching

    The cache lifetime is 5 minutes by default. If this is too short, you can apply one hour caching by enabling the `"extended-cache-ttl-2025-04-11"` beta header:

    ```python
    llm = ChatOllama(
        model="claude-3-7-sonnet-20250219",
        # highlight-next-line
        betas=["extended-cache-ttl-2025-04-11"],
    )
    ```
    and specifying `"cache_control": {"type": "ephemeral", "ttl": "1h"}`.

    Details of cached token counts will be included on the `InputTokenDetails` of response's `usage_metadata`:

    ```python
    response = llm.invoke(messages)
    response.usage_metadata
    ```
    ```
    {
        "input_tokens": 1500,
        "output_tokens": 200,
        "total_tokens": 1700,
        "input_token_details": {
            "cache_read": 0,
            "cache_creation": 1000,
            "ephemeral_1h_input_tokens": 750,
            "ephemeral_5m_input_tokens": 250,
        }
    }
    ```

:::

### Tools
"""
logger.info("# highlight-next-line")


description = (
    f"Get the weather at a location. By the way, check out this readme: {readme}"
)


@tool(description=description)
def get_weather(location: str) -> str:
    return "It's sunny."


weather_tool = convert_to_anthropic_tool(get_weather)
weather_tool["cache_control"] = {"type": "ephemeral"}

llm = ChatOllama(model="llama3.2")
llm_with_tools = llm.bind_tools([weather_tool])
query = "What's the weather in San Francisco?"

response_1 = llm_with_tools.invoke(query)
response_2 = llm_with_tools.invoke(query)

usage_1 = response_1.usage_metadata["input_token_details"]
usage_2 = response_2.usage_metadata["input_token_details"]

logger.debug(f"First invocation:\n{usage_1}")
logger.debug(f"\nSecond:\n{usage_2}")

"""
### Incremental caching in conversational applications

Prompt caching can be used in [multi-turn conversations](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#continuing-a-multi-turn-conversation) to maintain context from earlier messages without redundant processing.

We can enable incremental caching by marking the final message with `cache_control`. Claude will automatically use the longest previously-cached prefix for follow-up messages.

Below, we implement a simple chatbot that incorporates this feature. We follow the LangChain [chatbot tutorial](/docs/tutorials/chatbot/), but add a custom [reducer](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers) that automatically marks the last content block in each user message with `cache_control`. See below:
"""
logger.info("### Incremental caching in conversational applications")


llm = ChatOllama(model="llama3.2")

get_response = requests.get(
    "https://raw.githubusercontent.com/langchain-ai/langchain/master/README.md"
)
readme = get_response.text


def messages_reducer(left: list, right: list) -> list:
    for i in range(len(right) - 1, -1, -1):
        if right[i].type == "human":
            right[i].content[-1]["cache_control"] = {"type": "ephemeral"}
            break

    return add_messages(left, right)


class State(TypedDict):
    messages: Annotated[list, messages_reducer]


workflow = StateGraph(state_schema=State)


def call_model(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


config = {"configurable": {"thread_id": "abc123"}}

query = "Hi! I'm Bob."

input_message = HumanMessage([{"type": "text", "text": query}])
output = app.invoke({"messages": [input_message]}, config)
output["messages"][-1].pretty_logger.debug()
logger.debug(
    f"\n{output['messages'][-1].usage_metadata['input_token_details']}")

query = f"Check out this readme: {readme}"

input_message = HumanMessage([{"type": "text", "text": query}])
output = app.invoke({"messages": [input_message]}, config)
output["messages"][-1].pretty_logger.debug()
logger.debug(
    f"\n{output['messages'][-1].usage_metadata['input_token_details']}")

query = "What was my name again?"

input_message = HumanMessage([{"type": "text", "text": query}])
output = app.invoke({"messages": [input_message]}, config)
output["messages"][-1].pretty_logger.debug()
logger.debug(
    f"\n{output['messages'][-1].usage_metadata['input_token_details']}")

"""
In the [LangSmith trace](https://smith.langchain.com/public/4d0584d8-5f9e-4b91-8704-93ba2ccf416a/r), toggling "raw output" will show exactly what messages are sent to the chat model, including `cache_control` keys.

## Token-efficient tool use

Ollama supports a (beta) [token-efficient tool use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/token-efficient-tool-use) feature. To use it, specify the relevant beta-headers when instantiating the model.
"""
logger.info("## Token-efficient tool use")


llm = ChatOllama(
    model="claude-3-7-sonnet-20250219",
    temperature=0,
    model_kwargs={
        "extra_headers": {"anthropic-beta": "token-efficient-tools-2025-02-19"}
    },
)


@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return "It's sunny."


llm_with_tools = llm.bind_tools([get_weather])
response = llm_with_tools.invoke("What's the weather in San Francisco?")
logger.debug(response.tool_calls)
logger.debug(f"\nTotal tokens: {response.usage_metadata['total_tokens']}")

"""
## Citations

Ollama supports a [citations](https://docs.anthropic.com/en/docs/build-with-claude/citations) feature that lets Claude attach context to its answers based on source documents supplied by the user. When [document](https://docs.anthropic.com/en/docs/build-with-claude/citations#document-types) or `search result` content blocks with `"citations": {"enabled": True}` are included in a query, Claude may generate citations in its response.

### Simple example

In this example we pass a [plain text document](https://docs.anthropic.com/en/docs/build-with-claude/citations#plain-text-documents). In the background, Claude [automatically chunks](https://docs.anthropic.com/en/docs/build-with-claude/citations#plain-text-documents) the input text into sentences, which are used when generating citations.
"""
logger.info("## Citations")


llm = ChatOllama(model="llama3.2")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": "The grass is green. The sky is blue.",
                },
                "title": "My Document",
                "context": "This is a trustworthy document.",
                "citations": {"enabled": True},
            },
            {"type": "text", "text": "What color is the grass and sky?"},
        ],
    }
]
response = llm.invoke(messages)
response.content

"""
### In tool results (agentic RAG)

:::info Requires ``langchain-anthropic>=0.3.17``

:::

Claude supports a [search_result](https://docs.anthropic.com/en/docs/build-with-claude/search-results) content block representing citable results from queries against a knowledge base or other custom source. These content blocks can be passed to claude both top-line (as in the above example) and within a tool result. This allows Claude to cite elements of its response using the result of a tool call.

To pass search results in response to tool calls, define a tool that returns a list of `search_result` content blocks in Ollama's native format. For example:
```python
def retrieval_tool(query: str) -> list[dict]:
    """
logger.info("### In tool results (agentic RAG)")Access my knowledge base."""

    # Run a search (e.g., with a LangChain vector store)
    results = vector_store.similarity_search(query=query, k=2)

    # Package results into search_result blocks
    return [
        {
            "type": "search_result",
            # Customize fields as desired, using document metadata or otherwise
            "title": "My Document Title",
            "source": "Source description or provenance",
            "citations": {"enabled": True},
            "content": [{"type": "text", "text": doc.page_content}],
        }
        for doc in results
    ]
```

We also need to specify the `search-results-2025-06-09` beta when instantiating ChatOllama. You can see an end-to-end example below.

<details>
<summary>End to end example with LangGraph</summary>

Here we demonstrate an end-to-end example in which we populate a LangChain [vector store](/docs/concepts/vectorstores/) with sample documents and equip Claude with a tool that queries those documents.
The tool here takes a search query and a `category` string literal, but any valid tool signature can be used.

```python



# Set up vector store
embeddings = init_embeddings("ollama:nomic-embed-text")
vector_store = InMemoryVectorStore(embeddings)

document_1 = Document(
    id="1",
    page_content=(
        "To request vacation days, submit a leave request form through the "
        "HR portal. Approval will be sent by email."
    ),
    metadata={
        "category": "HR Policy",
        "doc_title": "Leave Policy",
        "provenance": "Leave Policy - page 1",
    },
)
document_2 = Document(
    id="2",
    page_content="Managers will review vacation requests within 3 business days.",
    metadata={
        "category": "HR Policy",
        "doc_title": "Leave Policy",
        "provenance": "Leave Policy - page 2",
    },
)
document_3 = Document(
    id="3",
    page_content=(
        "Employees with over 6 months tenure are eligible for 20 paid vacation days "
        "per year."
    ),
    metadata={
        "category": "Benefits Policy",
        "doc_title": "Benefits Guide 2025",
        "provenance": "Benefits Policy - page 1",
    },
)

documents = [document_1, document_2, document_3]
vector_store.add_documents(documents=documents)


# Define tool
async def retrieval_tool(
    query: str, category: Literal["HR Policy", "Benefits Policy"]
) -> list[dict]:
    """
logger.info("# Run a search (e.g., with a LangChain vector store)")Access my knowledge base."""

    def _filter_function(doc: Document) -> bool:
        return doc.metadata.get("category") == category

    results = vector_store.similarity_search(
        query=query, k=2, filter=_filter_function
    )

    return [
        {
            "type": "search_result",
            "title": doc.metadata["doc_title"],
            "source": doc.metadata["provenance"],
            "citations": {"enabled": True},
            "content": [{"type": "text", "text": doc.page_content}],
        }
        for doc in results
    ]



# Create agent
llm = init_chat_model(
    "ollama:claude-3-5-haiku-latest",
    betas=["search-results-2025-06-09"],
)

checkpointer = InMemorySaver()
agent = create_react_agent(llm, [retrieval_tool], checkpointer=checkpointer)


# Invoke on a query
config = {"configurable": {"thread_id": "session_1"}}

input_message = {
    "role": "user",
    "content": "How do I request vacation days?",
}
for step in agent.stream(
    {"messages": [input_message]},
    config,
    stream_mode="values",
):
    step["messages"][-1].pretty_logger.debug()
```

</details>

### Using with text splitters

Ollama also lets you specify your own splits using [custom document](https://docs.anthropic.com/en/docs/build-with-claude/citations#custom-content-documents) types. LangChain [text splitters](/docs/concepts/text_splitters/) can be used to generate meaningful splits for this purpose. See the below example, where we split the LangChain README (a markdown document) and pass it to Claude as context:
"""
logger.info("# Create agent")


def format_to_anthropic_documents(documents: list[str]):
    return {
        "type": "document",
        "source": {
            "type": "content",
            "content": [{"type": "text", "text": document} for document in documents],
        },
        "citations": {"enabled": True},
    }


get_response = requests.get(
    "https://raw.githubusercontent.com/langchain-ai/langchain/master/README.md"
)
readme = get_response.text

splitter = MarkdownTextSplitter(
    chunk_overlap=0,
    chunk_size=50,
)
documents = splitter.split_text(readme)

message = {
    "role": "user",
    "content": [
        format_to_anthropic_documents(documents),
        {"type": "text", "text": "Give me a link to LangChain's tutorials."},
    ],
}

llm = ChatOllama(model="llama3.2")
response = llm.invoke([message])

response.content

"""
## Built-in tools

Ollama supports a variety of [built-in tools](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/text-editor-tool), which can be bound to the model in the [usual way](/docs/how_to/tool_calling/). Claude will generate tool calls adhering to its internal schema for the tool:

### Web search

Claude can use a [web search tool](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-search-tool) to run searches and ground its responses with citations.

:::info Web search tool is supported since ``langchain-anthropic>=0.3.13``

:::
"""
logger.info("## Built-in tools")


llm = ChatOllama(model="llama3.2")

tool = {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}
llm_with_tools = llm.bind_tools([tool])

response = llm_with_tools.invoke(
    "How do I update a web app to TypeScript 5.5?")

"""
#### Web search + structured output

When combining web search tools with structured output, it's important to **bind the tools first and then apply structured output**:
"""
logger.info("#### Web search + structured output")


class ResearchResult(BaseModel):
    """Structured research result from web search."""

    topic: str = Field(description="The research topic")
    summary: str = Field(description="Summary of key findings")
    key_points: list[str] = Field(
        description="List of important points discovered")


websearch_tools = [
    {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 10,
    }
]

llm = ChatOllama(model="llama3.2")

llm_with_search = llm.bind_tools(websearch_tools)
research_llm = llm_with_search.with_structured_output(ResearchResult)

result = research_llm.invoke(
    "Research the latest developments in quantum computing")
logger.debug(f"Topic: {result.topic}")
logger.debug(f"Summary: {result.summary}")
logger.debug(f"Key Points: {result.key_points}")

"""
### Code execution

Claude can use a [code execution tool](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/code-execution-tool) to execute Python code in a sandboxed environment.

:::info Code execution is supported since ``langchain-anthropic>=0.3.14``

:::
"""
logger.info("### Code execution")


llm = ChatOllama(
    model="claude-sonnet-4-20250514",
    betas=["code-execution-2025-05-22"],
)

tool = {"type": "code_execution_20250522", "name": "code_execution"}
llm_with_tools = llm.bind_tools([tool])

response = llm_with_tools.invoke(
    "Calculate the mean and standard deviation of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"
)

"""
<details>
<summary>Use with Files API</summary>

Using the Files API, Claude can write code to access files for data analysis and other purposes. See example below:

```python
# Upload file


client = anthropic.Ollama()
file = client.beta.files.upload(
    file=open("/path/to/sample_data.csv", "rb")
)
file_id = file.id


# Run inference

llm = ChatOllama(
    model="claude-sonnet-4-20250514",
    betas=["code-execution-2025-05-22"],
)

tool = {"type": "code_execution_20250522", "name": "code_execution"}
llm_with_tools = llm.bind_tools([tool])

input_message = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "Please plot these data and tell me what you see.",
        },
        {
            "type": "container_upload",
            "file_id": file_id,
        },
    ]
}
llm_with_tools.invoke([input_message])
```

Note that Claude may generate files as part of its code execution. You can access these files using the Files API:
```python
# Take all file outputs for demonstration purposes
file_ids = []
for block in response.content:
    if block["type"] == "code_execution_tool_result":
        file_ids.extend(
            content["file_id"]
            for content in block.get("content", {}).get("content", [])
            if "file_id" in content
        )

for i, file_id in enumerate(file_ids):
    file_content = client.beta.files.download(file_id)
    file_content.write_to_file(f"/path/to/file_{i}.png")
```

</details>

### Remote MCP

Claude can use a [MCP connector tool](https://docs.anthropic.com/en/docs/agents-and-tools/mcp-connector) for model-generated calls to remote MCP servers.

:::info Remote MCP is supported since ``langchain-anthropic>=0.3.14``

:::
"""
logger.info("# Upload file")


mcp_servers = [
    {
        "type": "url",
        "url": "https://mcp.deepwiki.com/mcp",
        "name": "deepwiki",
        "tool_configuration": {  # optional configuration
            "enabled": True,
            "allowed_tools": ["ask_question"],
        },
        "authorization_token": "PLACEHOLDER",  # optional authorization
    }
]

llm = ChatOllama(
    model="claude-sonnet-4-20250514",
    betas=["mcp-client-2025-04-04"],
    mcp_servers=mcp_servers,
)

response = llm.invoke(
    "What transport protocols does the 2025-03-26 version of the MCP "
    "spec (modelcontextprotocol/modelcontextprotocol) support?"
)

"""
### Text editor

The text editor tool can be used to view and modify text files. See docs [here](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/text-editor-tool) for details.
"""
logger.info("### Text editor")


llm = ChatOllama(model="llama3.2")

tool = {"type": "text_editor_20250124", "name": "str_replace_editor"}
llm_with_tools = llm.bind_tools([tool])

response = llm_with_tools.invoke(
    "There's a syntax error in my primes.py file. Can you help me fix it?"
)
logger.debug(response.text())
response.tool_calls

"""
## API reference

For detailed documentation of all ChatOllama features and configurations head to the API reference: https://python.langchain.com/api_reference/anthropic/chat_models/jet.adapters.langchain.chat_ollama.chat_models.ChatOllama.html
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)
