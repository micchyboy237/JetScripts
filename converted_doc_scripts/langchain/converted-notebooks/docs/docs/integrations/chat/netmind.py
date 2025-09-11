from jet.logger import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_netmind import ChatNetmind
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
sidebar_label: Netmind
---

# ChatNetmind

This will help you get started with Netmind [chat models](https://www.netmind.ai/). For detailed documentation of all ChatNetmind features and configurations head to the [API reference](https://github.com/protagolabs/langchain-netmind).

-  See https://www.netmind.ai/ for an example.

## Overview
### Integration details

| Class                                                                                        | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/) | Package downloads | Package latest |
|:---------------------------------------------------------------------------------------------| :--- |:-----:|:------------:|:--------------------------------------------------------------:| :---: | :---: |
| [ChatNetmind](https://python.langchain.com/api_reference/) | [langchain-netmind](https://python.langchain.com/api_reference/) |   ✅   |      ❌       |                               ❌                                | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-netmind?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-netmind?style=flat-square&label=%20) |

### Model features
| [Tool calling](../../how_to/tool_calling.ipynb) | [Structured output](../../how_to/structured_output.ipynb) | JSON mode | [Image input](../../how_to/multimodal_inputs.ipynb) | Audio input | Video input | [Token-level streaming](../../how_to/chat_streaming.ipynb) | Native async | [Token usage](../../how_to/chat_token_usage_tracking.ipynb) | [Logprobs](../../how_to/logprobs.ipynb) |
|:-----------------------------------------------:|:---------------------------------------------------------:|:---------:|:---------------------------------------------------:|:-----------:|:-----------:|:----------------------------------------------------------:|:------------:|:-----------------------------------------------------------:|:---------------------------------------:|
|                        ✅                        |                             ✅                             |     ✅     |                          ❌                          |      ❌      |      ❌      |                             ✅                              |      ✅       |                              ✅                              |                    ✅                    | 

## Setup

To access Netmind models you'll need to create a/an Netmind account, get an API key, and install the `langchain-netmind` integration package.

### Credentials

Head to https://www.netmind.ai/ to sign up to Netmind and generate an API key. Once you've done this set the NETMIND_API_KEY environment variable:
"""
logger.info("# ChatNetmind")

# import getpass

if not os.getenv("NETMIND_API_KEY"):
#     os.environ["NETMIND_API_KEY"] = getpass.getpass("Enter your Netmind API key: ")

"""
If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:
"""
logger.info("If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:")



"""
### Installation

The LangChain Netmind integration lives in the `langchain-netmind` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-netmind

"""
## Instantiation

Now we can instantiate our model object and generate chat completions:
"""
logger.info("## Instantiation")


llm = ChatNetmind(
    model="deepseek-ai/DeepSeek-V3",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

"""
#
#
 
I
n
v
o
c
a
t
i
o
n
"""
logger.info("#")

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

For detailed documentation of all ChatNetmind features and configurations head to the API reference:  
* [API reference](https://python.langchain.com/api_reference/)  
* [langchain-netmind](https://github.com/protagolabs/langchain-netmind)  
* [pypi](https://pypi.org/project/langchain-netmind/)
"""
logger.info("## API reference")


logger.info("\n\n[DONE]", bright=True)