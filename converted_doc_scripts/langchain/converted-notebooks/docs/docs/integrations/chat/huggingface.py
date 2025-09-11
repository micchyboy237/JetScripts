from jet.logger import logger
from langchain_core.messages import (
HumanMessage,
SystemMessage,
)
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import BitsAndBytesConfig
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
# ChatHuggingFace

This will help you get started with `langchain_huggingface` [chat models](/docs/concepts/chat_models). For detailed documentation of all `ChatHuggingFace` features and configurations head to the [API reference](https://python.langchain.com/api_reference/huggingface/chat_models/langchain_huggingface.chat_models.huggingface.ChatHuggingFace.html). For a list of models supported by Hugging Face check out [this page](https://huggingface.co/models).

## Overview
### Integration details

### Integration details

| Class | Package | Local | Serializable | JS support | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [ChatHuggingFace](https://python.langchain.com/api_reference/huggingface/chat_models/langchain_huggingface.chat_models.huggingface.ChatHuggingFace.html) | [langchain-huggingface](https://python.langchain.com/api_reference/huggingface/index.html) | ✅ | beta | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain_huggingface?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain_huggingface?style=flat-square&label=%20) |

### Model features
| [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
| :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
| ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | 

## Setup

To access Hugging Face models you'll need to create a Hugging Face account, get an API key, and install the `langchain-huggingface` integration package.

### Credentials

Generate a [Hugging Face Access Token](https://huggingface.co/docs/hub/security-tokens) and store it as an environment variable: `HUGGINGFACEHUB_API_TOKEN`.
"""
logger.info("# ChatHuggingFace")

# import getpass

if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
#     os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Enter your token: ")

"""
### Installation

| Class | Package | Local | Serializable | JS support | Package downloads | Package latest |
| :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
| [ChatHuggingFace](https://python.langchain.com/api_reference/huggingface/chat_models/langchain_huggingface.chat_models.huggingface.ChatHuggingFace.html) | [langchain-huggingface](https://python.langchain.com/api_reference/huggingface/index.html) | ✅ | ❌ | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain_huggingface?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain_huggingface?style=flat-square&label=%20) |

### Model features
| [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
| :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
| ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 

## Setup

To access `langchain_huggingface` models you'll need to create a `Hugging Face` account, get an API key, and install the `langchain-huggingface` integration package.

### Credentials

You'll need to have a [Hugging Face Access Token](https://huggingface.co/docs/hub/security-tokens) saved as an environment variable: `HUGGINGFACEHUB_API_TOKEN`.
"""
logger.info("### Installation")

# import getpass

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass(
    "Enter your Hugging Face API key: "
)

# %pip install --upgrade --quiet  langchain-huggingface text-generation transformers google-search-results numexpr langchainhub sentencepiece jinja2 bitsandbytes accelerate

"""
## Instantiation

You can instantiate a `ChatHuggingFace` model in two different ways, either from a `HuggingFaceEndpoint` or from a `HuggingFacePipeline`.

### `HuggingFaceEndpoint`
"""
logger.info("## Instantiation")


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  # let Hugging Face choose the best provider for you
)

chat_model = ChatHuggingFace(llm=llm)

"""
Now let's take advantage of [Inference Providers](https://huggingface.co/docs/inference-providers) to run the model on specific third-party providers
"""
logger.info("Now let's take advantage of [Inference Providers](https://huggingface.co/docs/inference-providers) to run the model on specific third-party providers")

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    provider="hyperbolic",  # set your provider here
)

chat_model = ChatHuggingFace(llm=llm)

"""
### `HuggingFacePipeline`
"""
logger.info("### `HuggingFacePipeline`")


llm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    ),
)

chat_model = ChatHuggingFace(llm=llm)

"""
### Instatiating with Quantization

To run a quantized version of your model, you can specify a `bitsandbytes` quantization config as follows:
"""
logger.info("### Instatiating with Quantization")


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
)

"""
and pass it to the `HuggingFacePipeline` as a part of its `model_kwargs`:
"""
logger.info("and pass it to the `HuggingFacePipeline` as a part of its `model_kwargs`:")

llm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        return_full_text=False,
    ),
    model_kwargs={"quantization_config": quantization_config},
)

chat_model = ChatHuggingFace(llm=llm)

"""
## Invocation
"""
logger.info("## Invocation")


messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(
        content="What happens when an unstoppable force meets an immovable object?"
    ),
]

ai_msg = chat_model.invoke(messages)

logger.debug(ai_msg.content)

"""
## API reference

For detailed documentation of all `ChatHuggingFace` features and configurations head to the API reference: https://python.langchain.com/api_reference/huggingface/chat_models/langchain_huggingface.chat_models.huggingface.ChatHuggingFace.html

## API reference

For detailed documentation of all ChatHuggingFace features and configurations head to the API reference: https://python.langchain.com/api_reference/huggingface/chat_models/langchain_huggingface.chat_models.huggingface.ChatHuggingFace.html
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)