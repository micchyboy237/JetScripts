from IPython.display import Image
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama import ChatOllama, custom_tool
from jet.logger import logger
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
from pydantic import BaseModel, Field
import base64
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
sidebar_label: Ollama
---

# ChatOllama

This notebook provides a quick overview for getting started with Ollama [chat models](/docs/concepts/chat_models). For detailed documentation of all ChatOllama features and configurations head to the [API reference](https://python.langchain.com/api_reference/ollama/chat_models/jet.adapters.langchain.chat_ollama.chat_models.base.ChatOllama.html).

Ollama has several chat models. You can find information about their latest models and their costs, context windows, and supported input types in the [Ollama docs](https://platform.ollama.com/docs/models).

:::info Azure Ollama

Note that certain Ollama models can also be accessed via the [Microsoft Azure platform](https://azure.microsoft.com/en-us/products/ai-services/ollama-service). To use the Azure Ollama service use the [AzureChatOllama integration](/docs/integrations/chat/azure_chat_ollama/).

:::

## Overview

### Integration details
| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/ollama) | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [ChatOllama](https://python.langchain.com/api_reference/ollama/chat_models/jet.adapters.langchain.chat_ollama.chat_models.base.ChatOllama.html) | [langchain-ollama](https://python.langchain.com/api_reference/ollama/index.html) | ❌ | beta | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-ollama?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-ollama?style=flat-square&label=%20) |

### Model features
| [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | Image input | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
| :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
| ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | 

## Setup

To access Ollama models you'll need to create an Ollama account, get an API key, and install the `langchain-ollama` integration package.

### Credentials

# Head to https://platform.ollama.com to sign up to Ollama and generate an API key. Once you've done this set the OPENAI_API_KEY environment variable:
"""
logger.info("# ChatOllama")

# import getpass

# if not os.environ.get("OPENAI_API_KEY"):
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your Ollama API key: ")

"""
If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:
"""
logger.info("If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:")



"""
### Installation

The LangChain Ollama integration lives in the `langchain-ollama` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-ollama

"""
## Instantiation

Now we can instantiate our model object and generate chat completions:
"""
logger.info("## Instantiation")


llm = ChatOllama(
    model="llama3.2",
    temperature=0,
    max_tokens=None,
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
## Tool calling

Ollama has a [tool calling](https://platform.ollama.com/docs/guides/function-calling) (we use "tool calling" and "function calling" interchangeably here) API that lets you describe tools and their arguments, and have the model return a JSON object with a tool to invoke and the inputs to that tool. tool-calling is extremely useful for building tool-using chains and agents, and for getting structured outputs from models more generally.

### ChatOllama.bind_tools()

With `ChatOllama.bind_tools`, we can easily pass in Pydantic classes, dict schemas, LangChain tools, or even functions as tools to the model. Under the hood these are converted to an Ollama tool schemas, which looks like:
```
{
    "name": "...",
    "description": "...",
    "parameters": {...}  # JSONSchema
}
```
and passed in every model invocation.
"""
logger.info("## Tool calling")



class GetWeather(BaseModel):
    """Get the current weather in a given location"""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


llm_with_tools = llm.bind_tools([GetWeather])

ai_msg = llm_with_tools.invoke(
    "what is the weather like in San Francisco",
)
ai_msg

"""
### ``strict=True``

:::info Requires ``langchain-ollama>=0.1.21``

:::

As of Aug 6, 2024, Ollama supports a `strict` argument when calling tools that will enforce that the tool argument schema is respected by the model. See more here: https://platform.ollama.com/docs/guides/function-calling

**Note**: If ``strict=True`` the tool definition will also be validated, and a subset of JSON schema are accepted. Crucially, schema cannot have optional args (those with default values). Read the full docs on what types of schema are supported here: https://platform.ollama.com/docs/guides/structured-outputs/supported-schemas.
"""
logger.info("### ``strict=True``")

llm_with_tools = llm.bind_tools([GetWeather], strict=True)
ai_msg = llm_with_tools.invoke(
    "what is the weather like in San Francisco",
)
ai_msg

"""
### AIMessage.tool_calls
Notice that the AIMessage has a `tool_calls` attribute. This contains in a standardized ToolCall format that is model-provider agnostic.
"""
logger.info("### AIMessage.tool_calls")

ai_msg.tool_calls

"""
For more on binding tools and tool call outputs, head to the [tool calling](/docs/how_to/function_calling) docs.

### Structured output and tool calls

Ollama's [structured output](https://platform.ollama.com/docs/guides/structured-outputs) feature can be used simultaneously with tool-calling. The model will either generate tool calls or a response adhering to a desired schema. See example below:
"""
logger.info("### Structured output and tool calls")



def get_weather(location: str) -> None:
    """Get weather at a location."""
    return "It's sunny."


class OutputSchema(BaseModel):
    """Schema for response."""

    answer: str
    justification: str


llm = ChatOllama(model="llama3.2")

structured_llm = llm.bind_tools(
    [get_weather],
    response_format=OutputSchema,
    strict=True,
)

tool_call_response = structured_llm.invoke("What is the weather in SF?")

structured_response = structured_llm.invoke(
    "What weighs more, a pound of feathers or a pound of gold?"
)

"""
### Custom tools

:::info Requires ``langchain-ollama>=0.3.29``

:::

[Custom tools](https://platform.ollama.com/docs/guides/function-calling#custom-tools) support tools with arbitrary string inputs. They can be particularly useful when you expect your string arguments to be long or complex.
"""
logger.info("### Custom tools")



@custom_tool
def execute_code(code: str) -> str:
    """Execute python code."""
    return "27"


llm = ChatOllama(model="llama3.2")

agent = create_react_agent(llm, [execute_code])

input_message = {"role": "user", "content": "Use the tool to calculate 3^3."}
for step in agent.stream(
    {"messages": [input_message]},
    stream_mode="values",
):
    step["messages"][-1].pretty_logger.debug()

"""
<details>
<summary>Context-free grammars</summary>

Ollama supports the specification of a [context-free grammar](https://platform.ollama.com/docs/guides/function-calling#context-free-grammars) for custom tool inputs in `lark` or `regex` format. See [Ollama docs](https://platform.ollama.com/docs/guides/function-calling#context-free-grammars) for details. The `format` parameter can be passed into `@custom_tool` as shown below:
"""
logger.info("Ollama supports the specification of a [context-free grammar](https://platform.ollama.com/docs/guides/function-calling#context-free-grammars) for custom tool inputs in `lark` or `regex` format. See [Ollama docs](https://platform.ollama.com/docs/guides/function-calling#context-free-grammars) for details. The `format` parameter can be passed into `@custom_tool` as shown below:")


grammar = """
start: expr
expr: term (SP ADD SP term)* -> add
| term
term: factor (SP MUL SP factor)* -> mul
| factor
factor: INT
SP: " "
ADD: "+"
MUL: "*"
# %import common.INT
"""

format_ = {"type": "grammar", "syntax": "lark", "definition": grammar}


@custom_tool(format=format_)
def do_math(input_string: str) -> str:
    """Do a mathematical operation."""
    return "27"


llm = ChatOllama(model="llama3.2")

agent = create_react_agent(llm, [do_math])

input_message = {"role": "user", "content": "Use the tool to calculate 3^3."}
for step in agent.stream(
    {"messages": [input_message]},
    stream_mode="values",
):
    step["messages"][-1].pretty_logger.debug()

"""
</details>

## Responses API

:::info Requires ``langchain-ollama>=0.3.9``

:::

Ollama supports a [Responses](https://platform.ollama.com/docs/guides/responses-vs-chat-completions) API that is oriented toward building [agentic](/docs/concepts/agents/) applications. It includes a suite of [built-in tools](https://platform.ollama.com/docs/guides/tools?api-mode=responses), including web and file search. It also supports management of [conversation state](https://platform.ollama.com/docs/guides/conversation-state?api-mode=responses), allowing you to continue a conversational thread without explicitly passing in previous messages, as well as the output from [reasoning processes](https://platform.ollama.com/docs/guides/reasoning?api-mode=responses).

`ChatOllama` will route to the Responses API if one of these features is used. You can also specify `use_responses_api=True` when instantiating `ChatOllama`.

:::note

`langchain-ollama >= 0.3.26` allows users to opt-in to an updated AIMessage format when using the Responses API. Setting

```python
llm = ChatOllama(model="llama3.2")
```
will format output from reasoning summaries, built-in tool invocations, and other response items into the message's `content` field, rather than `additional_kwargs`. We recommend this format for new applications.

:::

### Web search

To trigger a web search, pass `{"type": "web_search_preview"}` to the model as you would another tool.

:::tip

You can also pass built-in tools as invocation params:
```python
llm.invoke("...", tools=[{"type": "web_search_preview"}])
```

:::
"""
logger.info("## Responses API")


llm = ChatOllama(model="llama3.2")

tool = {"type": "web_search_preview"}
llm_with_tools = llm.bind_tools([tool])

response = llm_with_tools.invoke("What was a positive news story from today?")

"""
Note that the response includes structured [content blocks](/docs/concepts/messages/#content-1) that include both the text of the response and Ollama [annotations](https://platform.ollama.com/docs/guides/tools-web-search?api-mode=responses#output-and-citations) citing its sources. The output message will also contain information from any tool invocations:
"""
logger.info("Note that the response includes structured [content blocks](/docs/concepts/messages/#content-1) that include both the text of the response and Ollama [annotations](https://platform.ollama.com/docs/guides/tools-web-search?api-mode=responses#output-and-citations) citing its sources. The output message will also contain information from any tool invocations:")

response.content

"""
:::tip

You can recover just the text content of the response as a string by using `response.text()`. For example, to stream response text:

```python
for token in llm_with_tools.stream("..."):
    logger.debug(token.text(), end="|")
```

See the [streaming guide](/docs/how_to/chat_streaming/) for more detail.

:::

### Image generation


:::info Requires ``langchain-ollama>=0.3.19``
:::


To trigger an image generation, pass `{"type": "image_generation"}` to the model as you would another tool.

:::tip

You can also pass built-in tools as invocation params:
```python
llm.invoke("...", tools=[{"type": "image_generation"}])
```

:::
"""
logger.info("### Image generation")


llm = ChatOllama(model="llama3.2")

tool = {"type": "image_generation", "quality": "low"}

llm_with_tools = llm.bind_tools([tool])

ai_message = llm_with_tools.invoke(
    "Draw a picture of a cute fuzzy cat with an umbrella"
)



image = next(
    item for item in ai_message.content if item["type"] == "image_generation_call"
)
Image(base64.b64decode(image["result"]), width=200)

"""
### File search

To trigger a file search, pass a [file search tool](https://platform.ollama.com/docs/guides/tools-file-search) to the model as you would another tool. You will need to populate an Ollama-managed vector store and include the vector store ID in the tool definition. See [Ollama documentation](https://platform.ollama.com/docs/guides/tools-file-search) for more detail.
"""
logger.info("### File search")


llm = ChatOllama(model="llama3.2")

ollama_vector_store_ids = [
    "vs_...",  # your IDs here
]

tool = {
    "type": "file_search",
    "vector_store_ids": ollama_vector_store_ids,
}
llm_with_tools = llm.bind_tools([tool])

response = llm_with_tools.invoke("What is deep research by Ollama?")
logger.debug(response.text())

"""
As with [web search](#web-search), the response will include content blocks with citations:
"""
logger.info("As with [web search](#web-search), the response will include content blocks with citations:")

[block["type"] for block in response.content]

text_block = next(block for block in response.content if block["type"] == "text")

text_block["annotations"][:2]

"""
It will also include information from the built-in tool invocations:
"""
logger.info("It will also include information from the built-in tool invocations:")

response.content[0]

"""
### Computer use

`ChatOllama` supports the `"computer-use-preview"` model, which is a specialized model for the built-in computer use tool. To enable, pass a [computer use tool](https://platform.ollama.com/docs/guides/tools-computer-use) as you would pass another tool.

Currently, tool outputs for computer use are present in the message `content` field. To reply to the computer use tool call, construct a `ToolMessage` with `{"type": "computer_call_output"}` in its `additional_kwargs`. The content of the message will be a screenshot. Below, we demonstrate a simple example.

First, load two screenshots:
"""
logger.info("### Computer use")



def load_png_as_base64(file_path):
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode("utf-8")


screenshot_1_base64 = load_png_as_base64(
    "/path/to/screenshot_1.png"
)  # perhaps a screenshot of an application
screenshot_2_base64 = load_png_as_base64(
    "/path/to/screenshot_2.png"
)  # perhaps a screenshot of the Desktop


llm = ChatOllama(
    model="computer-use-preview",
    truncation="auto",
    output_version="responses/v1",
)

tool = {
    "type": "computer_use_preview",
    "display_width": 1024,
    "display_height": 768,
    "environment": "browser",
}
llm_with_tools = llm.bind_tools([tool])

input_message = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": (
                "Click the red X to close and reveal my Desktop. "
                "Proceed, no confirmation needed."
            ),
        },
        {
            "type": "input_image",
            "image_url": f"data:image/png;base64,{screenshot_1_base64}",
        },
    ],
}

response = llm_with_tools.invoke(
    [input_message],
    reasoning={
        "generate_summary": "concise",
    },
)

"""
The response will include a call to the computer-use tool in its `content`:
"""
logger.info("The response will include a call to the computer-use tool in its `content`:")

response.content

"""
We next construct a ToolMessage with these properties:

1. It has a `tool_call_id` matching the `call_id` from the computer-call.
2. It has `{"type": "computer_call_output"}` in its `additional_kwargs`.
3. Its content is either an `image_url` or an `input_image` output block (see [Ollama docs](https://platform.ollama.com/docs/guides/tools-computer-use#5-repeat) for formatting).
"""
logger.info("We next construct a ToolMessage with these properties:")


tool_call_id = next(
    item["call_id"] for item in response.content if item["type"] == "computer_call"
)

tool_message = ToolMessage(
    content=[
        {
            "type": "input_image",
            "image_url": f"data:image/png;base64,{screenshot_2_base64}",
        }
    ],
    tool_call_id=tool_call_id,
    additional_kwargs={"type": "computer_call_output"},
)

"""
We can now invoke the model again using the message history:
"""
logger.info("We can now invoke the model again using the message history:")

messages = [
    input_message,
    response,
    tool_message,
]

response_2 = llm_with_tools.invoke(
    messages,
    reasoning={
        "generate_summary": "concise",
    },
)

response_2.text()

"""
Instead of passing back the entire sequence, we can also use the [previous_response_id](#passing-previous_response_id):
"""
logger.info("Instead of passing back the entire sequence, we can also use the [previous_response_id](#passing-previous_response_id):")

previous_response_id = response.response_metadata["id"]

response_2 = llm_with_tools.invoke(
    [tool_message],
    previous_response_id=previous_response_id,
    reasoning={
        "generate_summary": "concise",
    },
)

response_2.text()

"""
### Code interpreter

Ollama implements a [code interpreter](https://platform.ollama.com/docs/guides/tools-code-interpreter) tool to support the sandboxed generation and execution of code.

Example use:
"""
logger.info("### Code interpreter")


llm = ChatOllama(model="llama3.2")

llm_with_tools = llm.bind_tools(
    [
        {
            "type": "code_interpreter",
            "container": {"type": "auto"},
        }
    ]
)
response = llm_with_tools.invoke(
    "Write and run code to answer the question: what is 3^3?"
)

"""
Note that the above command created a new container. We can also specify an existing container ID:
"""
logger.info("Note that the above command created a new container. We can also specify an existing container ID:")

code_interpreter_calls = [
    item for item in response.content if item["type"] == "code_interpreter_call"
]
assert len(code_interpreter_calls) == 1
container_id = code_interpreter_calls[0]["container_id"]

llm_with_tools = llm.bind_tools(
    [
        {
            "type": "code_interpreter",
            "container": container_id,
        }
    ]
)

"""
### Remote MCP

Ollama implements a [remote MCP](https://platform.ollama.com/docs/guides/tools-remote-mcp) tool that allows for model-generated calls to MCP servers.

Example use:
"""
logger.info("### Remote MCP")


llm = ChatOllama(model="llama3.2")

llm_with_tools = llm.bind_tools(
    [
        {
            "type": "mcp",
            "server_label": "deepwiki",
            "server_url": "https://mcp.deepwiki.com/mcp",
            "require_approval": "never",
        }
    ]
)
response = llm_with_tools.invoke(
    "What transport protocols does the 2025-03-26 version of the MCP "
    "spec (modelcontextprotocol/modelcontextprotocol) support?"
)

"""
<details>
<summary>MCP Approvals</summary>

Ollama will at times request approval before sharing data with a remote MCP server.

In the above command, we instructed the model to never require approval. We can also configure the model to always request approval, or to always request approval for specific tools:

```python
llm_with_tools = llm.bind_tools(
    [
        {
            "type": "mcp",
            "server_label": "deepwiki",
            "server_url": "https://mcp.deepwiki.com/mcp",
            "require_approval": {
                "always": {
                    "tool_names": ["read_wiki_structure"]
                }
            }
        }
    ]
)
response = llm_with_tools.invoke(
    "What transport protocols does the 2025-03-26 version of the MCP "
    "spec (modelcontextprotocol/modelcontextprotocol) support?"
)
```

Responses may then include blocks with type `"mcp_approval_request"`.

To submit approvals for an approval request, structure it into a content block in an input message:

```python
approval_message = {
    "role": "user",
    "content": [
        {
            "type": "mcp_approval_response",
            "approve": True,
            "approval_request_id": block["id"],
        }
        for block in response.content
        if block["type"] == "mcp_approval_request"
    ]
}

next_response = llm_with_tools.invoke(
    [approval_message],
    # continue existing thread
    previous_response_id=response.response_metadata["id"]
)
```

</details>

### Managing conversation state

The Responses API supports management of [conversation state](https://platform.ollama.com/docs/guides/conversation-state?api-mode=responses).

#### Manually manage state

You can manage the state manually or using [LangGraph](/docs/tutorials/chatbot/), as with other chat models:
"""
logger.info("# continue existing thread")


llm = ChatOllama(model="llama3.2")

tool = {"type": "web_search_preview"}
llm_with_tools = llm.bind_tools([tool])

first_query = "What was a positive news story from today?"
messages = [{"role": "user", "content": first_query}]

response = llm_with_tools.invoke(messages)
response_text = response.text()
logger.debug(f"{response_text[:100]}... {response_text[-100:]}")

second_query = (
    "Repeat my question back to me, as well as the last sentence of your answer."
)

messages.extend(
    [
        response,
        {"role": "user", "content": second_query},
    ]
)
second_response = llm_with_tools.invoke(messages)
logger.debug(second_response.text())

"""
:::tip

You can use [LangGraph](https://langchain-ai.github.io/langgraph/) to manage conversational threads for you in a variety of backends, including in-memory and Postgres. See [this tutorial](/docs/tutorials/chatbot/) to get started.

:::


#### Passing `previous_response_id`

When using the Responses API, LangChain messages will include an `"id"` field in its metadata. Passing this ID to subsequent invocations will continue the conversation. Note that this is [equivalent](https://platform.ollama.com/docs/guides/conversation-state?api-mode=responses#ollama-apis-for-conversation-state) to manually passing in messages from a billing perspective.
"""
logger.info("#### Passing `previous_response_id`")


llm = ChatOllama(
    model="llama3.2",
    output_version="responses/v1",
)
response = llm.invoke("Hi, I'm Bob.")
logger.debug(response.text())

second_response = llm.invoke(
    "What is my name?",
    previous_response_id=response.response_metadata["id"],
)
logger.debug(second_response.text())

"""
ChatOllama can also automatically specify `previous_response_id` using the last response in a message sequence:
"""
logger.info("ChatOllama can also automatically specify `previous_response_id` using the last response in a message sequence:")


llm = ChatOllama(
    model="llama3.2",
    output_version="responses/v1",
    use_previous_response_id=True,
)

"""
If we set `use_previous_response_id=True`, input messages up to the most recent response will be dropped from request payloads, and `previous_response_id` will be set using the ID of the most recent response.

That is,
```python
llm.invoke(
    [
        HumanMessage("Hello"),
        AIMessage("Hi there!", response_metadata={"id": "resp_123"}),
        HumanMessage("How are you?"),
    ]
)
```
is equivalent to:
```python
llm.invoke([HumanMessage("How are you?")], previous_response_id="resp_123")
```

### Reasoning output

Some Ollama models will generate separate text content illustrating their reasoning process. See Ollama's [reasoning documentation](https://platform.ollama.com/docs/guides/reasoning?api-mode=responses) for details.

Ollama can return a summary of the model's reasoning (although it doesn't expose the raw reasoning tokens). To configure `ChatOllama` to return this summary, specify the `reasoning` parameter. `ChatOllama` will automatically route to the Responses API if this parameter is set.
"""
logger.info("### Reasoning output")


reasoning = {
    "effort": "medium",  # 'low', 'medium', or 'high'
    "summary": "auto",  # 'detailed', 'auto', or None
}

llm = ChatOllama(model="llama3.2")
response = llm.invoke("What is 3^3?")

response.text()

for block in response.content:
    if block["type"] == "reasoning":
        for summary in block["summary"]:
            logger.debug(summary["text"])

"""
## Fine-tuning

You can call fine-tuned Ollama models by passing in your corresponding `modelName` parameter.

This generally takes the form of `ft:{OPENAI_MODEL_NAME}:{ORG_NAME}::{MODEL_ID}`. For example:
"""
logger.info("## Fine-tuning")

fine_tuned_model = ChatOllama(
    temperature=0, model_name="ft:gpt-3.5-turbo-0613:langchain::7qTVM5AR"
)

fine_tuned_model.invoke(messages)

"""
## Multimodal Inputs (images, PDFs, audio)

Ollama has models that support multimodal inputs. You can pass in images, PDFs, or audio to these models. For more information on how to do this in LangChain, head to the [multimodal inputs](/docs/how_to/multimodal_inputs) docs.

You can see the list of models that support different modalities in [Ollama's documentation](https://platform.ollama.com/docs/models).

For all modalities, LangChain supports both its [cross-provider standard](/docs/concepts/multimodality/#multimodality-in-chat-models) as well as Ollama's native content-block format.

To pass multimodal data into `ChatOllama`, create a [content block](/docs/concepts/messages/) containing the data and incorporate it into a message, e.g., as below:
```python
message = {
    "role": "user",
    "content": [
        {
            "type": "text",
            # Update prompt as desired
            "text": "Describe the (image / PDF / audio...)",
        },
        # highlight-next-line
        content_block,
    ],
}
```
See below for examples of content blocks.

<details>
<summary>Images</summary>

Refer to examples in the how-to guide [here](/docs/how_to/multimodal_inputs/#images).

URLs:
```python
# LangChain format
content_block = {
    "type": "image",
    "source_type": "url",
    "url": url_string,
}

# Ollama Chat Completions format
content_block = {
    "type": "image_url",
    "image_url": {"url": url_string},
}
```

In-line base64 data:
```python
# LangChain format
content_block = {
    "type": "image",
    "source_type": "base64",
    "data": base64_string,
    "mime_type": "image/jpeg",
}

# Ollama Chat Completions format
content_block = {
    "type": "image_url",
    "image_url": {
        "url": f"data:image/jpeg;base64,{base64_string}",
    },
}
```

</details>


<details>
<summary>PDFs</summary>

Note: Ollama requires file-names be specified for PDF inputs. When using LangChain's format, include the `filename` key.

Read more [here](/docs/how_to/multimodal_inputs/#example-ollama-file-names).

Refer to examples in the how-to guide [here](/docs/how_to/multimodal_inputs/#documents-pdf).

In-line base64 data:
```python
# LangChain format
content_block = {
    "type": "file",
    "source_type": "base64",
    "data": base64_string,
    "mime_type": "application/pdf",
    # highlight-next-line
    "filename": "my-file.pdf",
}

# Ollama Chat Completions format
content_block = {
    "type": "file",
    "file": {
        "filename": "my-file.pdf",
        "file_data": f"data:application/pdf;base64,{base64_string}",
    }
}
```

</details>


<details>
<summary>Audio</summary>

See [supported models](https://platform.ollama.com/docs/models), e.g., `"gpt-4o-audio-preview"`.

Refer to examples in the how-to guide [here](/docs/how_to/multimodal_inputs/#audio).

In-line base64 data:
```python
# LangChain format
content_block = {
    "type": "audio",
    "source_type": "base64",
    "mime_type": "audio/wav",  # or appropriate mime-type
    "data": base64_string,
}

# Ollama Chat Completions format
content_block = {
    "type": "input_audio",
    "input_audio": {"data": base64_string, "format": "wav"},
}
```

</details>

## Predicted output

:::info
Requires `langchain-ollama>=0.2.6`
:::

Some Ollama models (such as their `gpt-4o` and `llama3.2` series) support [Predicted Outputs](https://platform.ollama.com/docs/guides/latency-optimization#use-predicted-outputs), which allow you to pass in a known portion of the LLM's expected output ahead of time to reduce latency. This is useful for cases such as editing text or code, where only a small part of the model's output will change.

Here's an example:
"""
logger.info("## Multimodal Inputs (images, PDFs, audio)")

code = """
/// <summary>
/// Represents a user with a first name, last name, and username.
/// </summary>
public class User
{
    /// <summary>
    /// Gets or sets the user's first name.
    /// </summary>
    public string FirstName { get; set; }

    /// <summary>
    /// Gets or sets the user's last name.
    /// </summary>
    public string LastName { get; set; }

    /// <summary>
    /// Gets or sets the user's username.
    /// </summary>
    public string Username { get; set; }
}
"""

llm = ChatOllama(model="llama3.2")
query = (
    "Replace the Username property with an Email property. "
    "Respond only with code, and with no markdown formatting."
)
response = llm.invoke(
    [{"role": "user", "content": query}, {"role": "user", "content": code}],
    prediction={"type": "content", "content": code},
)
logger.debug(response.content)
logger.debug(response.response_metadata)

"""
Note that currently predictions are billed as additional tokens and may increase your usage and costs in exchange for this reduced latency.

## Audio Generation (Preview)

:::info
Requires `langchain-ollama>=0.2.3`
:::

Ollama has a new [audio generation feature](https://platform.ollama.com/docs/guides/audio?audio-generation-quickstart-example=audio-out) that allows you to use audio inputs and outputs with the `gpt-4o-audio-preview` model.
"""
logger.info("## Audio Generation (Preview)")


llm = ChatOllama(
    model="llama3.2",
    temperature=0,
    model_kwargs={
        "modalities": ["text", "audio"],
        "audio": {"voice": "alloy", "format": "wav"},
    },
)

output_message = llm.invoke(
    [
        ("human", "Are you made by Ollama? Just answer yes or no"),
    ]
)

"""
`output_message.additional_kwargs['audio']` will contain a dictionary like
```python
{
    'data': '<audio data b64-encoded',
    'expires_at': 1729268602,
    'id': 'audio_67127d6a44348190af62c1530ef0955a',
    'transcript': 'Yes.'
}
```
and the format will be what was passed in `model_kwargs['audio']['format']`.

We can also pass this message with audio data back to the model as part of a message history before ollama `expires_at` is reached.

:::note
Output audio is stored under the `audio` key in `AIMessage.additional_kwargs`, but input content blocks are typed with an `input_audio` type and key in `HumanMessage.content` lists. 

For more information, see Ollama's [audio docs](https://platform.ollama.com/docs/guides/audio).
:::
"""
logger.info("and the format will be what was passed in `model_kwargs['audio']['format']`.")

history = [
    ("human", "Are you made by Ollama? Just answer yes or no"),
    output_message,
    ("human", "And what is your name? Just give your name."),
]
second_output_message = llm.invoke(history)

"""
## Flex processing

Ollama offers a variety of [service tiers](https://platform.ollama.com/docs/guides/flex-processing). The "flex" tier offers cheaper pricing for requests, with the trade-off that responses may take longer and resources might not always be available. This approach is best suited for non-critical tasks, including model testing, data enhancement, or jobs that can be run asynchronously.

To use it, initialize the model with `service_tier="flex"`:
```python
llm = ChatOllama(model="llama3.2")
```

Note that this is a beta feature that is only available for a subset of models. See Ollama [docs](https://platform.ollama.com/docs/guides/flex-processing) for more detail.

## API reference

For detailed documentation of all ChatOllama features and configurations head to the [API reference](https://python.langchain.com/api_reference/ollama/chat_models/jet.adapters.langchain.chat_ollama.chat_models.base.ChatOllama.html).
"""
logger.info("## Flex processing")

logger.info("\n\n[DONE]", bright=True)