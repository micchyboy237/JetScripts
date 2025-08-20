import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.base.llms.types import (
ChatMessage,
MessageRole,
)
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.sambanovasystems import SambaNovaCloud
from llama_index.llms.sambanovasystems import SambaStudio
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# SambaNova Systems

In this notebook you will know how to install, setup and use the [SambaNova Cloud](https://cloud.sambanova.ai/) and [SambaStudio](https://docs.sambanova.ai/sambastudio/latest/sambastudio-intro.html) platforms. Take a look and try it yourself!

# SambaNova Cloud

[SambaNova Cloud](https://cloud.sambanova.ai/) is a high-performance inference service that delivers rapid and precise results. Customers can seamlessly leverage SambaNova technology to enhance their user experience by integrating FastAPI inference APIs with their applications. This service provides an easy-to-use REST interface for streaming the inference results. Users are able to customize the inference parameters and pass the ML model on to the service.

## Setup

To access SambaNova Cloud model you will need to create a [SambaNovaCloud](https://cloud.sambanova.ai/apis) account, get an API key, install the `llama-index-llms-sambanova` integration package, and install the `SSEClient` Package.
"""
logger.info("# SambaNova Systems")

# %pip install llama-index-llms-sambanovasystems
# %pip install sseclient-py

"""
### Credentials

Get an API Key from [cloud.sambanova.ai](https://cloud.sambanova.ai/apis) and add it to your environment variables:

``` bash
export SAMBANOVA_API_KEY="your-api-key-here"
```

If you don't have it in your env variables, you can also add it in the pop-up input text.
"""
logger.info("### Credentials")

# import getpass

if not os.getenv("SAMBANOVA_API_KEY"):
#     os.environ["SAMBANOVA_API_KEY"] = getpass.getpass(
        "Enter your SambaNova Cloud API key: "
    )

"""
## Instantiation

Now we can instantiate our model object and generate chat completions:
"""
logger.info("## Instantiation")


llm = SambaNovaCloud(
    model="Meta-Llama-3.1-70B-Instruct",
    context_window=100000,
    max_tokens=1024,
    temperature=0.7,
    top_k=1,
    top_p=0.01,
)

"""
## Invocation

Given the following system and user messages, let's explore different ways of calling a SambaNova Cloud model.
"""
logger.info("## Invocation")


system_msg = ChatMessage(
    role=MessageRole.SYSTEM,
    content="You are a helpful assistant that translates English to French. Translate the user sentence.",
)
user_msg = ChatMessage(role=MessageRole.USER, content="I love programming.")

messages = [
    system_msg,
    user_msg,
]

"""
### Chat
"""
logger.info("### Chat")

ai_msg = llm.chat(messages)
ai_msg.message

logger.debug(ai_msg.message.content)

"""
### Complete
"""
logger.info("### Complete")

ai_msg = llm.complete(user_msg.content)
ai_msg

logger.debug(ai_msg.text)

"""
## Streaming

### Chat
"""
logger.info("## Streaming")

ai_stream_msgs = []
for stream in llm.stream_chat(messages):
    ai_stream_msgs.append(stream)
ai_stream_msgs

logger.debug(ai_stream_msgs[-1])

"""
### Complete
"""
logger.info("### Complete")

ai_stream_msgs = []
for stream in llm.stream_complete(user_msg.content):
    ai_stream_msgs.append(stream)
ai_stream_msgs

logger.debug(ai_stream_msgs[-1])

"""
## Async

### Chat
"""
logger.info("## Async")

async def run_async_code_4b15ba9a():
    async def run_async_code_e89bb483():
        ai_msg = llm.chat(messages)
        return ai_msg
    ai_msg = asyncio.run(run_async_code_e89bb483())
    logger.success(format_json(ai_msg))
    return ai_msg
ai_msg = asyncio.run(run_async_code_4b15ba9a())
logger.success(format_json(ai_msg))
ai_msg

logger.debug(ai_msg.message.content)

"""
### Complete
"""
logger.info("### Complete")

async def run_async_code_a18e5ace():
    async def run_async_code_8e16651b():
        ai_msg = llm.complete(user_msg.content)
        return ai_msg
    ai_msg = asyncio.run(run_async_code_8e16651b())
    logger.success(format_json(ai_msg))
    return ai_msg
ai_msg = asyncio.run(run_async_code_a18e5ace())
logger.success(format_json(ai_msg))
ai_msg

logger.debug(ai_msg.text)

"""
## Async Streaming

Not supported yet. Coming soon!

# SambaStudio

[SambaStudio](https://docs.sambanova.ai/sambastudio/latest/sambastudio-intro.html) is a rich, GUI-based platform that provides the functionality to train, deploy, and manage models.

## Setup

To access SambaStudio models you will need to be a __SambaNova customer__, deploy an endpoint using the GUI or CLI, and use the URL and API Key to connect to the endpoint, as described in the [SambaStudio endpoint documentation](https://docs.sambanova.ai/sambastudio/latest/endpoints.html#_endpoint_api_keys). Then, install the `llama-index-llms-sambanova` integration package, and install the `SSEClient` Package.
"""
logger.info("## Async Streaming")

# %pip install llama-index-llms-sambanova
# %pip install sseclient-py

"""
### Credentials

An endpoint must be deployed in SambaStudio to get the URL and API Key. Once they're available, include them to your environment variables:

``` bash
export SAMBASTUDIO_URL="your-url-here"
export SAMBASTUDIO_API_KEY="your-api-key-here"
```
"""
logger.info("### Credentials")

# import getpass

if not os.getenv("SAMBASTUDIO_URL"):
#     os.environ["SAMBASTUDIO_URL"] = getpass.getpass(
        "Enter your SambaStudio endpoint's URL: "
    )

if not os.getenv("SAMBASTUDIO_API_KEY"):
#     os.environ["SAMBASTUDIO_API_KEY"] = getpass.getpass(
        "Enter your SambaStudio endpoint's API key: "
    )

"""
## Instantiation

Now we can instantiate our model object and generate chat completions:
"""
logger.info("## Instantiation")


llm = SambaStudio(
    model="Meta-Llama-3-70B-Instruct-4096",
    context_window=100000,
    max_tokens=1024,
    temperature=0.7,
    top_k=1,
    top_p=0.01,
)

"""
## Invocation

Given the following system and user messages, let's explore different ways of calling a SambaNova Cloud model.
"""
logger.info("## Invocation")


system_msg = ChatMessage(
    role=MessageRole.SYSTEM,
    content="You are a helpful assistant that translates English to French. Translate the user sentence.",
)
user_msg = ChatMessage(role=MessageRole.USER, content="I love programming.")

messages = [
    system_msg,
    user_msg,
]

"""
### Chat
"""
logger.info("### Chat")

ai_msg = llm.chat(messages)
ai_msg.message

logger.debug(ai_msg.message.content)

"""
### Complete
"""
logger.info("### Complete")

ai_msg = llm.complete(user_msg.content)
ai_msg

logger.debug(ai_msg.text)

"""
## Streaming

### Chat
"""
logger.info("## Streaming")

ai_stream_msgs = []
for stream in llm.stream_chat(messages):
    ai_stream_msgs.append(stream)
ai_stream_msgs

logger.debug(ai_stream_msgs[-1])

"""
### Complete
"""
logger.info("### Complete")

ai_stream_msgs = []
for stream in llm.stream_complete(user_msg.content):
    ai_stream_msgs.append(stream)
ai_stream_msgs

logger.debug(ai_stream_msgs[-1])

"""
## Async

### Chat
"""
logger.info("## Async")

async def run_async_code_4b15ba9a():
    async def run_async_code_e89bb483():
        ai_msg = llm.chat(messages)
        return ai_msg
    ai_msg = asyncio.run(run_async_code_e89bb483())
    logger.success(format_json(ai_msg))
    return ai_msg
ai_msg = asyncio.run(run_async_code_4b15ba9a())
logger.success(format_json(ai_msg))
ai_msg

logger.debug(ai_msg.message.content)

"""
### Complete
"""
logger.info("### Complete")

async def run_async_code_a18e5ace():
    async def run_async_code_8e16651b():
        ai_msg = llm.complete(user_msg.content)
        return ai_msg
    ai_msg = asyncio.run(run_async_code_8e16651b())
    logger.success(format_json(ai_msg))
    return ai_msg
ai_msg = asyncio.run(run_async_code_a18e5ace())
logger.success(format_json(ai_msg))
ai_msg

logger.debug(ai_msg.text)

"""
## Async Streaming

Not supported yet. Coming soon!
"""
logger.info("## Async Streaming")

logger.info("\n\n[DONE]", bright=True)