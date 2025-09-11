from jet.logger import logger
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_greennode import ChatGreenNode
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
sidebar_label: GreenNode
---

# ChatGreenNode

>[GreenNode](https://greennode.ai/) is a global AI solutions provider and a **NVIDIA Preferred Partner**, delivering full-stack AI capabilities—from infrastructure to application—for enterprises across the US, MENA, and APAC regions. Operating on **world-class infrastructure** (LEED Gold, TIA‑942, Uptime Tier III), GreenNode empowers enterprises, startups, and researchers with a comprehensive suite of AI services

This page will help you get started with GreenNode Serverless AI [chat models](../../concepts/chat_models.mdx). For detailed documentation of all ChatGreenNode features and configurations head to the [API reference](https://python.langchain.com/api_reference/greennode/chat_models/langchain_greennode.chat_models.ChatGreenNode.html).


[GreenNode AI](https://greennode.ai/) offers an API to query [20+ leading open-source models](https://aiplatform.console.greennode.ai/models)

## Overview
### Integration details

| Class | Package | Local | Serializable | JS support | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [ChatGreenNode](https://python.langchain.com/api_reference/greennode/chat_models/langchain_greennode.chat_models.ChatGreenNode.html) | [langchain-greennode](https://python.langchain.com/api_reference/greennode/index.html) | ❌ | beta | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-greennode?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-greennode?style=flat-square&label=%20) |

### Model features
| [Tool calling](../../how_to/tool_calling.ipynb) | [Structured output](../../how_to/structured_output.ipynb) | JSON mode | [Image input](../../how_to/multimodal_inputs.ipynb) | Audio input | Video input | [Token-level streaming](../../how_to/chat_streaming.ipynb) | Native async | [Token usage](../../how_to/chat_token_usage_tracking.ipynb) | [Logprobs](../../how_to/logprobs.ipynb) |
| :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
| ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |

## Setup

To access GreenNode models you'll need to create a GreenNode account, get an API key, and install the `langchain-greennode` integration package.

### Credentials

Head to [this page](https://aiplatform.console.greennode.ai/api-keys) to sign up to GreenNode AI Platform and generate an API key. Once you've done this, set the GREENNODE_API_KEY environment variable:
"""
logger.info("# ChatGreenNode")

# import getpass

if not os.getenv("GREENNODE_API_KEY"):
#     os.environ["GREENNODE_API_KEY"] = getpass.getpass("Enter your GreenNode API key: ")

"""
If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:
"""
logger.info("If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:")



"""
### Installation

The LangChain GreenNode integration lives in the `langchain-greennode` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-greennode

"""
## Instantiation

Now we can instantiate our model object and generate chat completions:
"""
logger.info("## Instantiation")


llm = ChatGreenNode(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",  # Choose from available models
    temperature=0.6,
    top_p=0.95,
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
### Streaming

You can also stream the response using the `stream` method:
"""
logger.info("### Streaming")

for chunk in llm.stream("Write a short poem about artificial intelligence"):
    logger.debug(chunk.content, end="", flush=True)

"""
### Chat Messages

You can use different message types to structure your conversations with the model:
"""
logger.info("### Chat Messages")


messages = [
    SystemMessage(content="You are a helpful AI assistant with expertise in science."),
    HumanMessage(content="What are black holes?"),
    AIMessage(
        content="Black holes are regions of spacetime where gravity is so strong that nothing, including light, can escape from them."
    ),
    HumanMessage(content="How are they formed?"),
]

response = llm.invoke(messages)
logger.debug(response.content[:100])

"""
## Chaining

You can use `ChatGreenNode` in LangChain chains and agents:
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
## Available Models

The full list of supported models can be found in the [GreenNode Serverless AI Models](https://greennode.ai/product/model-as-a-service).

## API reference

For more details about the GreenNode Serverless AI API, visit the [GreenNode Serverless AI Documentation](https://helpdesk.greennode.ai/portal/en/kb/articles/greennode-maas-api).
"""
logger.info("## Available Models")

logger.info("\n\n[DONE]", bright=True)