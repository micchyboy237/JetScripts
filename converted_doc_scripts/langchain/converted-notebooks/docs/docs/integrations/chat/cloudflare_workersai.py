from jet.logger import logger
from langchain_cloudflare.chat_models import ChatCloudflareWorkersAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from typing import List
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
sidebar_label: CloudflareWorkersAI
---

# ChatCloudflareWorkersAI


This will help you get started with CloudflareWorkersAI [chat models](/docs/concepts/chat_models). For detailed documentation of all ChatCloudflareWorkersAI features and configurations head to the [API reference](https://python.langchain.com/docs/integrations/chat/cloudflare_workersai/).


## Overview
### Integration details


| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/cloudflare) | Package downloads | Package latest |
| :--- | :--- |:-----:|:------------:|:------------------------------------------------------------------------:| :---: | :---: |
| [ChatCloudflareWorkersAI](https://python.langchain.com/docs/integrations/chat/cloudflare_workersai/) | [langchain-cloudflare](https://pypi.org/project/langchain-cloudflare/) |   ✅   |      ❌       |                                     ❌                                     | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-cloudflare?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-cloudflare?style=flat-square&label=%20) |

### Model features
| [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
|:-----------------------------------------:|:----------------------------------------------------:|:---------:|:----------------------------------------------:|:-----------:|:-----------:|:-----------------------------------------------------:|:------------:|:------------------------------------------------------:|:----------------------------------:|
|                     ✅                     |                          ✅                           |     ✅     |                       ❌                        |      ❌      |      ❌      |                           ❌                           |      ❌       |                             ✅                            |                 ❌                  | 

## Setup

To access CloudflareWorkersAI models you'll need to create a/an CloudflareWorkersAI account, get an API key, and install the `langchain-cloudflare` integration package.

### Credentials


Head to https://www.cloudflare.com/developer-platform/products/workers-ai/ to sign up to CloudflareWorkersAI and generate an API key. Once you've done this set the CF_AI_API_KEY environment variable and the CF_ACCOUNT_ID environment variable:
"""
logger.info("# ChatCloudflareWorkersAI")

# import getpass

if not os.getenv("CF_AI_API_KEY"):
#     os.environ["CF_AI_API_KEY"] = getpass.getpass(
        "Enter your CloudflareWorkersAI API key: "
    )

if not os.getenv("CF_ACCOUNT_ID"):
#     os.environ["CF_ACCOUNT_ID"] = getpass.getpass(
        "Enter your CloudflareWorkersAI account ID: "
    )

"""
If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:
"""
logger.info("If you want to get automated tracing of your model calls you can also set your [LangSmith](https://docs.smith.langchain.com/) API key by uncommenting below:")



"""
### Installation

The LangChain CloudflareWorkersAI integration lives in the `langchain-cloudflare` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-cloudflare

"""
## Instantiation

Now we can instantiate our model object and generate chat completions:

- Update model instantiation with relevant params.
"""
logger.info("## Instantiation")


llm = ChatCloudflareWorkersAI(
    model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    temperature=0,
    max_tokens=1024,
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
## Structured Outputs
"""
logger.info("## Structured Outputs")

json_schema = {
    "title": "joke",
    "description": "Joke to tell user.",
    "type": "object",
    "properties": {
        "setup": {
            "type": "string",
            "description": "The setup of the joke",
        },
        "punchline": {
            "type": "string",
            "description": "The punchline to the joke",
        },
        "rating": {
            "type": "integer",
            "description": "How funny the joke is, from 1 to 10",
            "default": None,
        },
    },
    "required": ["setup", "punchline"],
}
structured_llm = llm.with_structured_output(json_schema)

structured_llm.invoke("Tell me a joke about cats")

"""
## Bind tools
"""
logger.info("## Bind tools")




@tool
def validate_user(user_id: int, addresses: List[str]) -> bool:
    """Validate user using historical addresses.

    Args:
        user_id (int): the user ID.
        addresses (List[str]): Previous addresses as a list of strings.
    """
    return True


llm_with_tools = llm.bind_tools([validate_user])

result = llm_with_tools.invoke(
    "Could you validate user 123? They previously lived at "
    "123 Fake St in Boston MA and 234 Pretend Boulevard in "
    "Houston TX."
)
result.tool_calls

"""
## API reference

https://developers.cloudflare.com/workers-ai/
https://developers.cloudflare.com/agents/
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)