from jet.logger import logger
from langchain_core.prompts import PromptTemplate
from langchain_fireworks import Fireworks
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
# Fireworks

:::caution
You are currently on a page documenting the use of Fireworks models as [text completion models](/docs/concepts/text_llms). Many popular Fireworks models are [chat completion models](/docs/concepts/chat_models).

You may be looking for [this page instead](/docs/integrations/chat/fireworks/).
:::

>[Fireworks](https://app.fireworks.ai/) accelerates product development on generative AI by creating an innovative AI experiment and production platform. 

This example goes over how to use LangChain to interact with `Fireworks` models.

## Overview
### Integration details

| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/v0.1/docs/integrations/llms/fireworks/) | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [Fireworks](https://python.langchain.com/api_reference/fireworks/llms/langchain_fireworks.llms.Fireworks.html#langchain_fireworks.llms.Fireworks) | [langchain-fireworks](https://python.langchain.com/api_reference/fireworks/index.html) | ❌ | ❌ | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain_fireworks?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain_fireworks?style=flat-square&label=%20) |

## Setup

### Credentials 

Sign in to [Fireworks AI](http://fireworks.ai) for the an API Key to access our models, and make sure it is set as the `FIREWORKS_API_KEY` environment variable.
3. Set up your model using a model id. If the model is not set, the default model is fireworks-llama-v2-7b-chat. See the full, most up-to-date model list on [fireworks.ai](https://fireworks.ai).
"""
logger.info("# Fireworks")

# import getpass

if "FIREWORKS_API_KEY" not in os.environ:
#     os.environ["FIREWORKS_API_KEY"] = getpass.getpass("Fireworks API Key:")

"""
### Installation

You need to install the `langchain-fireworks` python package for the rest of the notebook to work.
"""
logger.info("### Installation")

# %pip install -qU langchain-fireworks

"""
## Instantiation
"""
logger.info("## Instantiation")


llm = Fireworks(
    model="accounts/fireworks/models/llama-v3p1-8b-instruct",
    base_url="https://api.fireworks.ai/inference/v1/completions",
)

"""
## Invocation

You can call the model directly with string prompts to get completions.
"""
logger.info("## Invocation")

output = llm.invoke("Who's the best quarterback in the NFL?")
logger.debug(output)

"""
### Invoking with multiple prompts
"""
logger.info("### Invoking with multiple prompts")

output = llm.generate(
    [
        "Who's the best cricket player in 2016?",
        "Who's the best basketball player in the league?",
    ]
)
logger.debug(output.generations)

"""
### Invoking with additional parameters
"""
logger.info("### Invoking with additional parameters")

llm = Fireworks(
    model="accounts/fireworks/models/llama-v3p1-8b-instruct",
    temperature=0.7,
    max_tokens=15,
    top_p=1.0,
)
logger.debug(llm.invoke("What's the weather like in Kansas City in December?"))

"""
## Chaining

You can use the LangChain Expression Language to create a simple chain with non-chat models.
"""
logger.info("## Chaining")


llm = Fireworks(
    model="accounts/fireworks/models/llama-v3p1-8b-instruct",
    temperature=0.7,
    max_tokens=15,
    top_p=1.0,
)
prompt = PromptTemplate.from_template("Tell me a joke about {topic}?")
chain = prompt | llm

logger.debug(chain.invoke({"topic": "bears"}))

"""
## Streaming

You can stream the output, if you want.
"""
logger.info("## Streaming")

for token in chain.stream({"topic": "bears"}):
    logger.debug(token, end="", flush=True)

"""
## API reference

For detailed documentation of all `Fireworks` LLM features and configurations head to the API reference: https://python.langchain.com/api_reference/fireworks/llms/langchain_fireworks.llms.Fireworks.html#langchain_fireworks.llms.Fireworks
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)