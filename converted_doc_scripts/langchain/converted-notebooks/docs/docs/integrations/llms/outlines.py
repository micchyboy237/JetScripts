from jet.logger import logger
from langchain_community.llms import Outlines
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
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
# Outlines

This will help you get started with Outlines LLM. For detailed documentation of all Outlines features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/llms/langchain_community.llms.outlines.Outlines.html).

[Outlines](https://github.com/outlines-dev/outlines) is a library for constrained language generation. It allows you to use large language models (LLMs) with various backends while applying constraints to the generated output.

## Overview

### Integration details
| Class | Package | Local | Serializable | JS support | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [Outlines](https://python.langchain.com/api_reference/community/llms/langchain_community.llms.outlines.Outlines.html) | [langchain-community](https://python.langchain.com/api_reference/community/index.html) | ✅ | beta | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-community?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-community?style=flat-square&label=%20) |

## Setup

To access Outlines models you'll need to have an internet connection to download the model weights from huggingface. Depending on the backend you need to install the required dependencies (see [Outlines docs](https://dottxt-ai.github.io/outlines/latest/installation/))

### Credentials

There is no built-in auth mechanism for Outlines.

## Installation

The LangChain Outlines integration lives in the `langchain-community` package and requires the `outlines` library:
"""
logger.info("# Outlines")

# %pip install -qU langchain-community outlines

"""
## Instantiation

Now we can instantiate our model object and generate chat completions:
"""
logger.info("## Instantiation")


model = Outlines(model="microsoft/Phi-3-mini-4k-instruct", backend="llamacpp")

model = Outlines(model="microsoft/Phi-3-mini-4k-instruct", backend="vllm")

model = Outlines(model="microsoft/Phi-3-mini-4k-instruct", backend="mlxlm")

model = Outlines(
    model="microsoft/Phi-3-mini-4k-instruct"
)  # defaults to backend="transformers"

"""
## Invocation
"""
logger.info("## Invocation")

model.invoke("Hello how are you?")

"""
## Chaining
"""
logger.info("## Chaining")


prompt = PromptTemplate.from_template("How to say {input} in {output_language}:\n")

chain = prompt | model
chain.invoke(
    {
        "output_language": "German",
        "input": "I love programming.",
    }
)

"""
### Streaming

Outlines supports streaming of tokens:
"""
logger.info("### Streaming")

for chunk in model.stream("Count to 10 in French:"):
    logger.debug(chunk, end="", flush=True)

"""
### Constrained Generation

Outlines allows you to apply various constraints to the generated output:

#### Regex Constraint
"""
logger.info("### Constrained Generation")

model.regex = r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)"
response = model.invoke("What is the IP address of Google's DNS server?")

response

"""
### Type Constraints
"""
logger.info("### Type Constraints")

model.type_constraints = int
response = model.invoke("What is the answer to life, the universe, and everything?")

"""
#### JSON Schema
"""
logger.info("#### JSON Schema")



class Person(BaseModel):
    name: str


model.json_schema = Person
response = model.invoke("Who is the author of LangChain?")
person = Person.model_validate_json(response)

person

"""
#### Grammar Constraint
"""
logger.info("#### Grammar Constraint")

model.grammar = """
?start: expression
?expression: term (("+" | "-") term)
?term: factor (("" | "/") factor)
?factor: NUMBER | "-" factor | "(" expression ")"
# %import common.NUMBER
# %import common.WS
# %ignore WS
"""
response = model.invoke("Give me a complex arithmetic expression:")

response

"""
## API reference

For detailed documentation of all ChatOutlines features and configurations head to the API reference: https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.outlines.ChatOutlines.html

## Outlines Documentation: 

https://dottxt-ai.github.io/outlines/latest/
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)