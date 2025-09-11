from jet.logger import logger
from langchain_abso import ChatAbso
from langchain_core.prompts import ChatPromptTemplate
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
sidebar_label: Abso
---

# ChatAbso

This will help you get started with ChatAbso [chat models](https://python.langchain.com/docs/concepts/chat_models/). For detailed documentation of all ChatAbso features and configurations, head to the [API reference](https://python.langchain.com/api_reference/en/latest/chat_models/langchain_abso.chat_models.ChatAbso.html).

- You can find the full documentation for the Abso router [here] (https://abso.ai)

## Overview
### Integration details

| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/abso) | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [ChatAbso](https://python.langchain.com/api_reference/en/latest/chat_models/langchain_abso.chat_models.ChatAbso.html) | [langchain-abso](https://python.langchain.com/api_reference/en/latest/abso_api_reference.html) | ❌ | ❌ | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-abso?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-abso?style=flat-square&label=%20) |

## Setup
To access ChatAbso models, you'll need to create an Ollama account, get an API key, and install the `langchain-abso` integration package.

### Credentials

- TODO: Update with relevant info.

Head to (TODO: link) to sign up for ChatAbso and generate an API key. Once you've done this, set the ABSO_API_KEY environment variable:
"""
logger.info("# ChatAbso")

# import getpass

# if not os.getenv("OPENAI_API_KEY"):
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your Ollama API key: ")

"""
### Installation

The LangChain ChatAbso integration lives in the `langchain-abso` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-abso

"""
## Instantiation

Now we can instantiate our model object and generate chat completions:
"""
logger.info("## Instantiation")


llm = ChatAbso(fast_model="llama3.2", slow_model="o3-mini")

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

For detailed documentation of all ChatAbso features and configurations head to the API reference: https://python.langchain.com/api_reference/en/latest/chat_models/langchain_abso.chat_models.ChatAbso.html
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)