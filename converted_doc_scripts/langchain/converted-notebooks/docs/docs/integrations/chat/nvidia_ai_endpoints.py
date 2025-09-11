from jet.logger import logger
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pydantic import Field
import IPython
import base64
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
sidebar_label: NVIDIA AI Endpoints
---

# ChatNVIDIA

This will help you get started with NVIDIA [chat models](/docs/concepts/chat_models). For detailed documentation of all `ChatNVIDIA` features and configurations head to the [API reference](https://python.langchain.com/api_reference/nvidia_ai_endpoints/chat_models/langchain_nvidia_ai_endpoints.chat_models.ChatNVIDIA.html).

## Overview
The `langchain-nvidia-ai-endpoints` package contains LangChain integrations building applications with models on
NVIDIA NIM inference microservice. NIM supports models across domains like chat, embedding, and re-ranking models
from the community as well as NVIDIA. These models are optimized by NVIDIA to deliver the best performance on NVIDIA
accelerated infrastructure and deployed as a NIM, an easy-to-use, prebuilt containers that deploy anywhere using a single
command on NVIDIA accelerated infrastructure.

NVIDIA hosted deployments of NIMs are available to test on the [NVIDIA API catalog](https://build.nvidia.com/). After testing,
NIMs can be exported from NVIDIA’s API catalog using the NVIDIA AI Enterprise license and run on-premises or in the cloud,
giving enterprises ownership and full control of their IP and AI application.

NIMs are packaged as container images on a per model basis and are distributed as NGC container images through the NVIDIA NGC Catalog.
At their core, NIMs provide easy, consistent, and familiar APIs for running inference on an AI model.

This example goes over how to use LangChain to interact with NVIDIA supported via the `ChatNVIDIA` class.

For more information on accessing the chat models through this api, check out the [ChatNVIDIA](https://python.langchain.com/docs/integrations/chat/nvidia_ai_endpoints/) documentation.

### Integration details

| Class | Package | Local | Serializable | JS support | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [ChatNVIDIA](https://python.langchain.com/api_reference/nvidia_ai_endpoints/chat_models/langchain_nvidia_ai_endpoints.chat_models.ChatNVIDIA.html) | [langchain-nvidia-ai-endpoints](https://python.langchain.com/api_reference/nvidia_ai_endpoints/index.html) | ✅ | beta | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain_nvidia_ai_endpoints?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain_nvidia_ai_endpoints?style=flat-square&label=%20) |

### Model features
| [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
| :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
| ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |

## Setup

**To get started:**

1. Create a free account with [NVIDIA](https://build.nvidia.com/), which hosts NVIDIA AI Foundation models.

2. Click on your model of choice.

3. Under `Input` select the `Python` tab, and click `Get API Key`. Then click `Generate Key`.

4. Copy and save the generated key as `NVIDIA_API_KEY`. From there, you should have access to the endpoints.

### Credentials
"""
logger.info("# ChatNVIDIA")

# import getpass

if not os.getenv("NVIDIA_API_KEY"):
#     os.environ["NVIDIA_API_KEY"] = getpass.getpass("Enter your NVIDIA API key: ")

"""
T
o
 
e
n
a
b
l
e
 
a
u
t
o
m
a
t
e
d
 
t
r
a
c
i
n
g
 
o
f
 
y
o
u
r
 
m
o
d
e
l
 
c
a
l
l
s
,
 
s
e
t
 
y
o
u
r
 
[
L
a
n
g
S
m
i
t
h
]
(
h
t
t
p
s
:
/
/
d
o
c
s
.
s
m
i
t
h
.
l
a
n
g
c
h
a
i
n
.
c
o
m
/
)
 
A
P
I
 
k
e
y
:
"""
logger.info("T")



"""
### Installation

The LangChain NVIDIA AI Endpoints integration lives in the `langchain-nvidia-ai-endpoints` package:
"""
logger.info("### Installation")

# %pip install --upgrade --quiet langchain-nvidia-ai-endpoints

"""
## Instantiation

Now we can access models in the NVIDIA API Catalog:
"""
logger.info("## Instantiation")


llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")

"""
## Invocation
"""
logger.info("## Invocation")

result = llm.invoke("Write a ballad about LangChain.")
logger.debug(result.content)

"""
## Working with NVIDIA NIMs
When ready to deploy, you can self-host models with NVIDIA NIM—which is included with the NVIDIA AI Enterprise software license—and run them anywhere, giving you ownership of your customizations and full control of your intellectual property (IP) and AI applications.

[Learn more about NIMs](https://developer.nvidia.com/blog/nvidia-nim-offers-optimized-inference-microservices-for-deploying-ai-models-at-scale/)
"""
logger.info("## Working with NVIDIA NIMs")


llm = ChatNVIDIA(base_url="http://localhost:8000/v1", model="meta/llama3-8b-instruct")

"""
## Stream, Batch, and Async

These models natively support streaming, and as is the case with all LangChain LLMs they expose a batch method to handle concurrent requests, as well as async methods for invoke, stream, and batch. Below are a few examples.
"""
logger.info("## Stream, Batch, and Async")

logger.debug(llm.batch(["What's 2*3?", "What's 2*6?"]))

for chunk in llm.stream("How far can a seagull fly in one day?"):
    logger.debug(chunk.content, end="|")

for chunk in llm.stream(
    "How long does it take for monarch butterflies to migrate?"
):
    logger.debug(chunk.content, end="|")

"""
## Supported models

Querying `available_models` will still give you all of the other models offered by your API credentials.

The `playground_` prefix is optional.
"""
logger.info("## Supported models")

ChatNVIDIA.get_available_models()

"""
## Model types

All of these models above are supported and can be accessed via `ChatNVIDIA`.

Some model types support unique prompting techniques and chat messages. We will review a few important ones below.

**To find out more about a specific model, please navigate to the API section of an AI Foundation model [as linked here](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-foundation/models/codellama-13b/api).**

### General Chat

Models such as `meta/llama3-8b-instruct` and `mistralai/mixtral-8x22b-instruct-v0.1` are good all-around models that you can use for with any LangChain chat messages. Example below.
"""
logger.info("## Model types")


prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful AI assistant named Fred."), ("user", "{input}")]
)
chain = prompt | ChatNVIDIA(model="meta/llama3-8b-instruct") | StrOutputParser()

for txt in chain.stream({"input": "What's your name?"}):
    logger.debug(txt, end="")

"""
### Code Generation

These models accept the same arguments and input structure as regular chat models, but they tend to perform better on code-generation and structured code tasks. An example of this is `meta/codellama-70b`.
"""
logger.info("### Code Generation")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert coding AI. Respond only in valid python; no narration whatsoever.",
        ),
        ("user", "{input}"),
    ]
)
chain = prompt | ChatNVIDIA(model="meta/codellama-70b") | StrOutputParser()

for txt in chain.stream({"input": "How do I solve this fizz buzz problem?"}):
    logger.debug(txt, end="")

"""
## Multimodal

NVIDIA also supports multimodal inputs, meaning you can provide both images and text for the model to reason over. An example model supporting multimodal inputs is `nvidia/neva-22b`.

Below is an example use:
"""
logger.info("## Multimodal")


image_url = "https://www.nvidia.com/content/dam/en-zz/Solutions/research/ai-playground/nvidia-picasso-3c33-p@2x.jpg"  ## Large Image
image_content = requests.get(image_url).content

IPython.display.Image(image_content)


llm = ChatNVIDIA(model="nvidia/neva-22b")

"""
#### Passing an image as a URL
"""
logger.info("#### Passing an image as a URL")


llm.invoke(
    [
        HumanMessage(
            content=[
                {"type": "text", "text": "Describe this image:"},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        )
    ]
)

"""
#### Passing an image as a base64 encoded string

A
t
 
t
h
e
 
m
o
m
e
n
t
,
 
s
o
m
e
 
e
x
t
r
a
 
p
r
o
c
e
s
s
i
n
g
 
h
a
p
p
e
n
s
 
c
l
i
e
n
t
-
s
i
d
e
 
t
o
 
s
u
p
p
o
r
t
 
l
a
r
g
e
r
 
i
m
a
g
e
s
 
l
i
k
e
 
t
h
e
 
o
n
e
 
a
b
o
v
e
.
 
B
u
t
 
f
o
r
 
s
m
a
l
l
e
r
 
i
m
a
g
e
s
 
(
a
n
d
 
t
o
 
b
e
t
t
e
r
 
i
l
l
u
s
t
r
a
t
e
 
t
h
e
 
p
r
o
c
e
s
s
 
g
o
i
n
g
 
o
n
 
u
n
d
e
r
 
t
h
e
 
h
o
o
d
)
,
 
w
e
 
c
a
n
 
d
i
r
e
c
t
l
y
 
p
a
s
s
 
i
n
 
t
h
e
 
i
m
a
g
e
 
a
s
 
s
h
o
w
n
 
b
e
l
o
w
:
"""
logger.info("#### Passing an image as a base64 encoded string")


image_url = "https://picsum.photos/seed/kitten/300/200"
image_content = requests.get(image_url).content

IPython.display.Image(image_content)



b64_string = base64.b64encode(image_content).decode("utf-8")

llm.invoke(
    [
        HumanMessage(
            content=[
                {"type": "text", "text": "Describe this image:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_string}"},
                },
            ]
        )
    ]
)

"""
#### Directly within the string

The NVIDIA API uniquely accepts images as base64 images inlined within `<img/>` HTML tags. While this isn't interoperable with other LLMs, you can directly prompt the model accordingly.
"""
logger.info("#### Directly within the string")

base64_with_mime_type = f"data:image/png;base64,{b64_string}"
llm.invoke(f'What\'s in this image?\n<img src="{base64_with_mime_type}" />')

"""
## Example usage within a RunnableWithMessageHistory

Like any other integration, ChatNVIDIA is fine to support chat utilities like RunnableWithMessageHistory which is analogous to using `ConversationChain`. Below, we show the [LangChain RunnableWithMessageHistory](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html) example applied to the `mistralai/mixtral-8x22b-instruct-v0.1` model.
"""
logger.info("## Example usage within a RunnableWithMessageHistory")

# %pip install --upgrade --quiet langchain


store = {}  # memory is maintained outside the chain


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


chat = ChatNVIDIA(
    model="mistralai/mixtral-8x22b-instruct-v0.1",
    temperature=0.1,
    max_tokens=100,
    top_p=1.0,
)

config = {"configurable": {"session_id": "1"}}

conversation = RunnableWithMessageHistory(
    chat,
    get_session_history,
)

conversation.invoke(
    "Hi I'm Srijan Dubey.",  # input or query
    config=config,
)

conversation.invoke(
    "I'm doing well! Just having a conversation with an AI.",
    config=config,
)

conversation.invoke(
    "Tell me about yourself.",
    config=config,
)

"""
## Tool calling

Starting in v0.2, `ChatNVIDIA` supports [bind_tools](https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.chat_models.BaseChatModel.html#langchain_core.language_models.chat_models.BaseChatModel.bind_tools).

`ChatNVIDIA` provides integration with the variety of models on [build.nvidia.com](https://build.nvidia.com) as well as local NIMs. Not all these models are trained for tool calling. Be sure to select a model that does have tool calling for your experimention and applications.

You can get a list of models that are known to support tool calling with,
"""
logger.info("## Tool calling")

tool_models = [
    model for model in ChatNVIDIA.get_available_models() if model.supports_tools
]
tool_models

"""
With a tool capable model,
"""
logger.info("With a tool capable model,")



@tool
def get_current_weather(
    location: str = Field(..., description="The location to get the weather for."),
):
    """Get the current weather for a location."""
    ...


llm = ChatNVIDIA(model=tool_models[0].id).bind_tools(tools=[get_current_weather])
response = llm.invoke("What is the weather in Boston?")
response.tool_calls

"""
See [How to use chat models to call tools](https://python.langchain.com/docs/how_to/tool_calling/) for additional examples.

## Chaining

We can [chain](/docs/how_to/sequence/) our model with a prompt template like so:
"""
logger.info("## Chaining")


prompt = ChatPromptTemplate(
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
## API reference

For detailed documentation of all `ChatNVIDIA` features and configurations head to the API reference: https://python.langchain.com/api_reference/nvidia_ai_endpoints/chat_models/langchain_nvidia_ai_endpoints.chat_models.ChatNVIDIA.html
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)