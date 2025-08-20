import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.nvidia import NVIDIA
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
# NVIDIA's LLM Text Completion API

Extending the NVIDIA class to support /completion API's for below models:

- bigcode/starcoder2-7b
- bigcode/starcoder2-15b

## Installation
"""
logger.info("# NVIDIA's LLM Text Completion API")

# !pip install --force-reinstall llama_index-llms-nvidia

"""
## Setup

**To get started:**

1. Create a free account with [NVIDIA](https://build.nvidia.com/), which hosts NVIDIA AI Foundation models.

2. Click on your model of choice.

3. Under Input select the Python tab, and click `Get API Key`. Then click `Generate Key`.

4. Copy and save the generated key as NVIDIA_API_KEY. From there, you should have access to the endpoints.
"""
logger.info("## Setup")

# !which python

# import getpass

if os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    logger.debug("Valid NVIDIA_API_KEY already in environment. Delete to reset")
else:
#     nvapi_key = getpass.getpass("NVAPI Key (starts with nvapi-): ")
    assert nvapi_key.startswith(
        "nvapi-"
    ), f"{nvapi_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvapi_key

os.environ["NVIDIA_API_KEY"]

# import nest_asyncio

# nest_asyncio.apply()

"""
## Working with NVIDIA API Catalog
#### Usage of `use_chat_completions` argument: 
Set None (default) to per-invocation decide on using /chat/completions vs /completions endpoints with query keyword arguments

- set False to universally use /completions endpoint
- set True to universally use /chat/completions endpoint
"""
logger.info("## Working with NVIDIA API Catalog")


llm = NVIDIA(model="bigcode/starcoder2-15b", use_chat_completions=False)

"""
### Available Models

`is_chat_model` can be used to get available text completion models
"""
logger.info("### Available Models")

logger.debug([model for model in llm.available_models if model.is_chat_model])

"""
## Working with NVIDIA NIMs

In addition to connecting to hosted [NVIDIA NIMs](https://ai.nvidia.com), this connector can be used to connect to local microservice instances. This helps you take your applications local when necessary.

For instructions on how to setup local microservice instances, see https://developer.nvidia.com/blog/nvidia-nim-offers-optimized-inference-microservices-for-deploying-ai-models-at-scale/
"""
logger.info("## Working with NVIDIA NIMs")


llm = NVIDIA(base_url="http://localhost:8080/v1")

"""
### Complete: `.complete()`

We can use `.complete()`/`.acomplete()` (which takes a string) to prompt a response from the selected model.

Let's use our default model for this task.
"""
logger.info("### Complete: `.complete()`")

logger.debug(llm.complete("# Function that does quicksort:"))

"""
As is expected by LlamaIndex - we get a `CompletionResponse` in response.

#### Async Complete: `.acomplete()`

There is also an async implementation which can be leveraged in the same way!
"""
logger.info("#### Async Complete: `.acomplete()`")

async def run_async_code_01833c5d():
    llm.complete("# Function that does quicksort:")
    return 
 = asyncio.run(run_async_code_01833c5d())
logger.success(format_json())

"""
#### Streaming
"""
logger.info("#### Streaming")

x = llm.stream_complete(prompt="# Reverse string in python:", max_tokens=512)

for t in x:
    logger.debug(t.delta, end="")

"""
#### Async Streaming
"""
logger.info("#### Async Streaming")

async def async_func_0():
    x = llm.stream_complete(
        prompt="# Reverse program in python:", max_tokens=512
    )
    return x
x = asyncio.run(async_func_0())
logger.success(format_json(x))

async for t in x:
    logger.debug(t.delta, end="")

logger.info("\n\n[DONE]", bright=True)