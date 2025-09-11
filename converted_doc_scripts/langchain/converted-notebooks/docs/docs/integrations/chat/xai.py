from jet.logger import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_xai import ChatXAI
from pydantic import BaseModel, Field
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
sidebar_label: xAI
---

# ChatXAI


This page will help you get started with xAI [chat models](../../concepts/chat_models.mdx). For detailed documentation of all `ChatXAI` features and configurations, head to the [API reference](https://python.langchain.com/api_reference/xai/chat_models/langchain_xai.chat_models.ChatXAI.html).

[xAI](https://console.x.ai/) offers an API to interact with Grok models.

## Overview
### Integration details

| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/xai) | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [ChatXAI](https://python.langchain.com/api_reference/xai/chat_models/langchain_xai.chat_models.ChatXAI.html) | [langchain-xai](https://python.langchain.com/api_reference/xai/index.html) | ❌ | beta | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-xai?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-xai?style=flat-square&label=%20) |

### Model features
| [Tool calling](../../how_to/tool_calling.ipynb) | [Structured output](../../how_to/structured_output.ipynb) | JSON mode | [Image input](../../how_to/multimodal_inputs.ipynb) | Audio input | Video input | [Token-level streaming](../../how_to/chat_streaming.ipynb) | Native async | [Token usage](../../how_to/chat_token_usage_tracking.ipynb) | [Logprobs](../../how_to/logprobs.ipynb) |
| :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
| ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ |

## Setup

To access xAI models, you'll need to create an xAI account, get an API key, and install the `langchain-xai` integration package.

### Credentials

Head to [this page](https://console.x.ai/) to sign up for xAI and generate an API key. Once you've done this, set the `XAI_API_KEY` environment variable:
"""
logger.info("# ChatXAI")

# import getpass

if "XAI_API_KEY" not in os.environ:
#     os.environ["XAI_API_KEY"] = getpass.getpass("Enter your xAI API key: ")

"""
To enable automated tracing of your model calls, set your [LangSmith](https://docs.smith.langchain.com/) API key:
"""
logger.info("To enable automated tracing of your model calls, set your [LangSmith](https://docs.smith.langchain.com/) API key:")



"""
### Installation

The LangChain xAI integration lives in the `langchain-xai` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-xai

"""
## Instantiation

Now we can instantiate our model object and generate chat completions:
"""
logger.info("## Instantiation")


llm = ChatXAI(
    model="grok-beta",
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

We can [chain](../../how_to/sequence.ipynb) our model with a prompt template like so:
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

ChatXAI has a [tool calling](https://docs.x.ai/docs#capabilities) (we use "tool calling" and "function calling" interchangeably here) API that lets you describe tools and their arguments, and have the model return a JSON object with a tool to invoke and the inputs to that tool. Tool-calling is extremely useful for building tool-using chains and agents, and for getting structured outputs from models more generally.

### ChatXAI.bind_tools()

With `ChatXAI.bind_tools`, we can easily pass in Pydantic classes, dict schemas, LangChain tools, or even functions as tools to the model. Under the hood, these are converted to an Ollama tool schema, which looks like:
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
## Live Search

xAI supports a [Live Search](https://docs.x.ai/docs/guides/live-search) feature that enables Grok to ground its answers using results from web searches:
"""
logger.info("## Live Search")


llm = ChatXAI(
    model="grok-3-latest",
    search_parameters={
        "mode": "auto",
        "max_search_results": 3,
        "from_date": "2025-05-26",
        "to_date": "2025-05-27",
    },
)

llm.invoke("Provide me a digest of world news in the last 24 hours.")

"""
See [xAI docs](https://docs.x.ai/docs/guides/live-search) for the full set of web search options.

## API reference

For detailed documentation of all `ChatXAI` features and configurations, head to the [API reference](https://python.langchain.com/api_reference/xai/chat_models/langchain_xai.chat_models.ChatXAI.html).
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)