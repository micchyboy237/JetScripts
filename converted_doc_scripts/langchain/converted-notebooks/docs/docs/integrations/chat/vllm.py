from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import (
ChatPromptTemplate,
HumanMessagePromptTemplate,
SystemMessagePromptTemplate,
)
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
sidebar_label: vLLM Chat
---

# vLLM Chat

vLLM can be deployed as a server that mimics the Ollama API protocol. This allows vLLM to be used as a drop-in replacement for applications using Ollama API. This server can be queried in the same format as Ollama API.

## Overview
This will help you get started with vLLM [chat models](/docs/concepts/chat_models), which leverages the `langchain-ollama` package. For detailed documentation of all `ChatOllama` features and configurations head to the [API reference](https://python.langchain.com/api_reference/ollama/chat_models/jet.adapters.langchain.chat_ollama.chat_models.base.ChatOllama.html).

### Integration details

| Class | Package | Local | Serializable | JS support | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [ChatOllama](https://python.langchain.com/api_reference/ollama/chat_models/jet.adapters.langchain.chat_ollama.chat_models.base.ChatOllama.html) | [jet.adapters.langchain.chat_ollama](https://python.langchain.com/api_reference/ollama/) | ✅ | beta | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/jet.adapters.langchain.chat_ollama?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/jet.adapters.langchain.chat_ollama?style=flat-square&label=%20) |

### Model features
Specific model features, such as tool calling, support for multi-modal inputs, support for token-level streaming, etc., will depend on the hosted model.

## Setup

See the vLLM docs [here](https://docs.vllm.ai/en/latest/).

To access vLLM models through LangChain, you'll need to install the `langchain-ollama` integration package.

### Credentials

Authentication will depend on specifics of the inference server.

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
logger.info("# vLLM Chat")



"""
### Installation

The LangChain vLLM integration can be accessed via the `langchain-ollama` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-ollama

"""
## Instantiation

Now we can instantiate our model object and generate chat completions:
"""
logger.info("## Instantiation")


inference_server_url = "http://localhost:8000/v1"

llm = ChatOllama(
    model="mosaicml/mpt-7b",
    ollama_ollama_api_base=inference_server_url,
    max_tokens=5,
    temperature=0,
)

"""
## Invocation
"""
logger.info("## Invocation")

messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to Italian."
    ),
    HumanMessage(
        content="Translate the following sentence from English to Italian: I love programming."
    ),
]
llm.invoke(messages)

"""
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

For detailed documentation of all features and configurations exposed via `langchain-ollama`, head to the API reference: https://python.langchain.com/api_reference/ollama/chat_models/jet.adapters.langchain.chat_ollama.chat_models.base.ChatOllama.html

Refer to the vLLM [documentation](https://docs.vllm.ai/en/latest/) as well.
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)