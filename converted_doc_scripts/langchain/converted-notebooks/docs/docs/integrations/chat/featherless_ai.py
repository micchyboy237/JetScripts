from jet.logger import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_featherless_ai import ChatFeatherlessAi
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
sidebar_label: Featherless AI
---

# ChatFeatherlessAi


This will help you get started with FeatherlessAi [chat models](/docs/concepts/chat_models). For detailed documentation of all ChatFeatherlessAi features and configurations head to the [API reference](https://python.langchain.com/api_reference/__package_name_short_snake__/chat_models/__module_name__.chat_models.ChatFeatherlessAi.html).

- See https://featherless.ai/ for an example.

## Overview
### Integration details


| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/__package_name_short_snake__) | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [ChatFeatherlessAi](https://python.langchain.com/api_reference/__package_name_short_snake__/chat_models/__module_name__.chat_models.ChatFeatherlessAi.html) | [langchain-featherless-ai](https://python.langchain.com/api_reference/__package_name_short_snake__/) | ✅ | ❌ | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-featherless-ai?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-featherless-ai?style=flat-square&label=%20) |

### Model features
| [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
| :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
| ❌ | ❌ | ✅| ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | 

## Setup


To access Featherless AI models you'll need to create a/an Featherless AI account, get an API key, and install the `langchain-featherless-ai` integration package.

### Credentials


Head to https://featherless.ai/ to sign up to FeatherlessAI and generate an API key. Once you've done this set the FEATHERLESSAI_API_KEY environment variable:
"""
logger.info("# ChatFeatherlessAi")

# import getpass

if not os.getenv("FEATHERLESSAI_API_KEY"):
#     os.environ["FEATHERLESSAI_API_KEY"] = getpass.getpass(
        "Enter your FeatherlessAI API key: "
    )

"""
If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:
"""
logger.info("If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:")



"""
### Installation

The LangChain FeatherlessAi integration lives in the `langchain-featherless-ai` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-featherless-ai

"""
## Instantiation

Now we can instantiate our model object and generate chat completions:
"""
logger.info("## Instantiation")


llm = ChatFeatherlessAi(
    model="featherless-ai/Qwerky-72B",
    temperature=0.9,
    max_tokens=None,
    timeout=None,
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

For detailed documentation of all ChatFeatherlessAi features and configurations head to the [API reference](https://python.langchain.com/api_reference/__package_name_short_snake__/chat_models/.chat_models.ChatFeatherlessAi.html)
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)