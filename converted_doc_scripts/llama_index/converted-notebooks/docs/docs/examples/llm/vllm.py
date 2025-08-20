import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llms import ChatMessage
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.vllm import Vllm
from llama_index.llms.vllm import VllmServer
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
# vLLM  

There's two modes of using vLLM local and remote. Let's start form the former one, which requeries CUDA environment available locally. 

### Install vLLM

`pip install vllm` <br>
or if you want to compile you can [compile from source](https://docs.vllm.ai/en/latest/getting_started/installation.html)

### Orca-7b Completion Example
"""
logger.info("# vLLM")

# %pip install llama-index-llms-vllm


os.environ["HF_HOME"] = "model/"


llm = Vllm(
    model="microsoft/Orca-2-7b",
    tensor_parallel_size=4,
    max_new_tokens=100,
    vllm_kwargs={"swap_space": 1, "gpu_memory_utilization": 0.5},
)

llm.complete(
    ["[INST]You are a helpful assistant[/INST] What is a black hole ?"]
)

"""
### LLama-2-7b Completion Example
"""
logger.info("### LLama-2-7b Completion Example")

llm = Vllm(
    model="codellama/CodeLlama-7b-hf",
    dtype="float16",
    tensor_parallel_size=4,
    temperature=0,
    max_new_tokens=100,
    vllm_kwargs={
        "swap_space": 1,
        "gpu_memory_utilization": 0.5,
        "max_model_len": 4096,
    },
)

llm.complete(["import socket\n\ndef ping_exponential_backoff(host: str):"])

"""
### Mistral chat 7b Completion Example
"""
logger.info("### Mistral chat 7b Completion Example")

llm = Vllm(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    dtype="float16",
    tensor_parallel_size=4,
    temperature=0,
    max_new_tokens=100,
    vllm_kwargs={
        "swap_space": 1,
        "gpu_memory_utilization": 0.5,
        "max_model_len": 4096,
    },
)

llm.complete([" What is a black hole ?"])

"""
# Calling vLLM via HTTP

In this mode there is no need to install `vllm` model nor have CUDA available locally. To setup the vLLM API you can follow the guide present [here](https://docs.vllm.ai/en/latest/serving/distributed_serving.html). 
Note: `llama-index-llms-vllm` module is a client for `vllm.entrypoints.api_server` which is only [a demo](https://github.com/vllm-project/vllm/blob/abfc4f3387c436d46d6701e9ba916de8f9ed9329/vllm/entrypoints/api_server.py#L2). <br>
If vLLM server is launched with `vllm.entrypoints.openai.api_server` as [MLX Compatible Server](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#openai-compatible-server)  or via [Docker](https://docs.vllm.ai/en/latest/serving/deploying_with_docker.html) you need `MLXLike` class from `llama-index-llms-ollama-like` [module](localai.ipynb#llamaindex-interaction)

### Completion Response
"""
logger.info("# Calling vLLM via HTTP")


llm = VllmServer(
    api_url="http://localhost:8000/generate", max_new_tokens=100, temperature=0
)

llm.complete("what is a black hole ?")

message = [ChatMessage(content="hello", role="user")]
llm.chat(message)

"""
### Streaming Response
"""
logger.info("### Streaming Response")

list(llm.stream_complete("what is a black hole"))[-1]

message = [ChatMessage(content="what is a black hole", role="user")]
[x for x in llm.stream_chat(message)][-1]

"""
### Async Response
"""
logger.info("### Async Response")

async def run_async_code_fda572e1():
    llm.complete("What is a black hole")
    return 
 = asyncio.run(run_async_code_fda572e1())
logger.success(format_json())

async def run_async_code_4c92d4fe():
    llm.chat(message)
    return 
 = asyncio.run(run_async_code_4c92d4fe())
logger.success(format_json())

async def run_async_code_a62dbd0e():
    [x for x in llm.stream_complete("what is a black hole")][-1]
    return 
 = asyncio.run(run_async_code_a62dbd0e())
logger.success(format_json())

async def run_async_code_49c50b89():
    [x for x in llm.stream_chat(message)][-1]
    return 
 = asyncio.run(run_async_code_49c50b89())
logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)