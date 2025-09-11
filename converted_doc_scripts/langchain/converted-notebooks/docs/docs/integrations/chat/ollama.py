from IPython.display import HTML, display
from PIL import Image
from io import BytesIO
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.messages import AIMessage
from langchain_core.messages import ChatMessage, HumanMessage
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from typing import List
import base64
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
sidebar_label: Ollama
---

# ChatOllama

[Ollama](https://ollama.com/) allows you to run open-source large language models, such as `gpt-oss`, locally.

`ollama` bundles model weights, configuration, and data into a single package, defined by a Modelfile.

It optimizes setup and configuration details, including GPU usage.

For a complete list of supported models and model variants, see the [Ollama model library](https://github.com/jmorganca/ollama#model-library).

## Overview
### Integration details

| Class | Package | Local | Serializable | [JS support](https://js.langchain.com/docs/integrations/chat/ollama) | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [ChatOllama](https://python.langchain.com/api_reference/ollama/chat_models/jet.adapters.langchain.chat_ollama.chat_models.ChatOllama.html#chatollama) | [langchain-ollama](https://python.langchain.com/api_reference/ollama/index.html) | ✅ | ❌ | ✅ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-ollama?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-ollama?style=flat-square&label=%20) |

### Model features
| [Tool calling](/docs/how_to/tool_calling/) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
| :---: |:----------------------------------------------------:| :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
| ✅ |                          ✅                           | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ |

## Setup

First, follow [these instructions](https://github.com/ollama/ollama?tab=readme-ov-file#ollama) to set up and run a local Ollama instance:

* [Download](https://ollama.ai/download) and install Ollama onto the available supported platforms (including Windows Subsystem for Linux aka WSL, macOS, and Linux)
    * macOS users can install via Homebrew with `brew install ollama` and start with `brew services start ollama`
* Fetch available LLM model via `ollama pull <name-of-model>`
    * View a list of available models via the [model library](https://ollama.ai/library)
    * e.g., `ollama pull gpt-oss:20b`
* This will download the default tagged version of the model. Typically, the default points to the latest, smallest sized-parameter model.

> On Mac, the models will be download to `~/.ollama/models`
>
> On Linux (or WSL), the models will be stored at `/usr/share/ollama/.ollama/models`

* Specify the exact version of the model of interest as such `ollama pull gpt-oss:20b` (View the [various tags for the `Vicuna`](https://ollama.ai/library/vicuna/tags) model in this instance)
* To view all pulled models, use `ollama list`
* To chat directly with a model from the command line, use `ollama run <name-of-model>`
* View the [Ollama documentation](https://github.com/ollama/ollama/blob/main/docs/README.md) for more commands. You can run `ollama help` in the terminal to see available commands.

To enable automated tracing of your model calls, set your [LangSmith](https://docs.smith.langchain.com/) API key:
"""
logger.info("# ChatOllama")



"""
### Installation

The LangChain Ollama integration lives in the `langchain-ollama` package:
"""
logger.info("### Installation")

# %pip install -qU langchain-ollama

"""
:::warning
Make sure you're using the latest Ollama version!
:::

Update by running:
"""
logger.info("Make sure you're using the latest Ollama version!")

# %pip install -U ollama

"""
## Instantiation

Now we can instantiate our model object and generate chat completions:
"""
logger.info("## Instantiation")


llm = ChatOllama(
    model="llama3.1",
    temperature=0,
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

We can use [tool calling](/docs/concepts/tool_calling/) with an LLM [that has been fine-tuned for tool use](https://ollama.com/search?&c=tools) such as `gpt-oss`:

```
ollama pull gpt-oss:20b
```

Details on creating custom tools are available in [this guide](/docs/how_to/custom_tools/). Below, we demonstrate how to create a tool using the `@tool` decorator on a normal python function.
"""
logger.info("## Tool calling")




@tool
def validate_user(user_id: int, addresses: List[str]) -> bool:
    """Validate user using historical addresses.

    Args:
        user_id (int): the user ID.
        addresses (List[str]): Previous addresses as a list of strings.
    """
    return True


llm = ChatOllama(
    model="gpt-oss:20b",
    validate_model_on_init=True,
    temperature=0,
).bind_tools([validate_user])

result = llm.invoke(
    "Could you validate user 123? They previously lived at "
    "123 Fake St in Boston MA and 234 Pretend Boulevard in "
    "Houston TX."
)

if isinstance(result, AIMessage) and result.tool_calls:
    logger.debug(result.tool_calls)

"""
## Multi-modal

Ollama has limited support for multi-modal LLMs, such as [gemma3](https://ollama.com/library/gemma3)

Be sure to update Ollama so that you have the most recent version to support multi-modal.
"""
logger.info("## Multi-modal")

# %pip install pillow




def convert_to_base64(pil_image):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """

    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # You can change the format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def plt_img_base64(img_base64):
    """
    Disply base64 encoded string as image

    :param img_base64:  Base64 string
    """
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    display(HTML(image_html))


file_path = "../../../static/img/ollama_example_img.jpg"
pil_image = Image.open(file_path)

image_b64 = convert_to_base64(pil_image)
plt_img_base64(image_b64)


llm = ChatOllama(model="llama3.2")


def prompt_func(data):
    text = data["text"]
    image = data["image"]

    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{image}",
    }

    content_parts = []

    text_part = {"type": "text", "text": text}

    content_parts.append(image_part)
    content_parts.append(text_part)

    return [HumanMessage(content=content_parts)]



chain = prompt_func | llm | StrOutputParser()

query_chain = chain.invoke(
    {"text": "What is the Dollar-based gross retention rate?", "image": image_b64}
)

logger.debug(query_chain)

"""
## Reasoning models and custom message roles

Some models, such as IBM's [Granite 3.2](https://ollama.com/library/granite3.2), support custom message roles to enable thinking processes.

To access Granite 3.2's thinking features, pass a message with a `"control"` role with content set to `"thinking"`. Because `"control"` is a non-standard message role, we can use a [ChatMessage](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.chat.ChatMessage.html) object to implement it:
"""
logger.info("## Reasoning models and custom message roles")


llm = ChatOllama(model="llama3.2")

messages = [
    ChatMessage(role="control", content="thinking"),
    HumanMessage("What is 3^3?"),
]

response = llm.invoke(messages)
logger.debug(response.content)

"""
Note that the model exposes its thought process in addition to its final response.

## API reference

For detailed documentation of all ChatOllama features and configurations head to the [API reference](https://python.langchain.com/api_reference/ollama/chat_models/jet.adapters.langchain.chat_ollama.chat_models.ChatOllama.html).
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)