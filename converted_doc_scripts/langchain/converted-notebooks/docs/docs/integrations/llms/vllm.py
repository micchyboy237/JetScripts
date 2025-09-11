from jet.logger import logger
from langchain.chains import LLMChain
from langchain_community.llms import VLLM
from langchain_community.llms import VLLMOllama
from langchain_core.prompts import PromptTemplate
from vllm.lora.request import LoRARequest
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
# vLLM

[vLLM](https://vllm.readthedocs.io/en/latest/index.html) is a fast and easy-to-use library for LLM inference and serving, offering:

* State-of-the-art serving throughput 
* Efficient management of attention key and value memory with PagedAttention
* Continuous batching of incoming requests
* Optimized CUDA kernels

This notebooks goes over how to use a LLM with langchain and vLLM.

To use, you should have the `vllm` python package installed.
"""
logger.info("# vLLM")

# %pip install --upgrade --quiet  vllm -q


llm = VLLM(
    model="mosaicml/mpt-7b",
    trust_remote_code=True,  # mandatory for hf models
    max_new_tokens=128,
    top_k=10,
    top_p=0.95,
    temperature=0.8,
)

logger.debug(llm.invoke("What is the capital of France ?"))

"""
## Integrate the model in an LLMChain
"""
logger.info("## Integrate the model in an LLMChain")


template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Who was the US president in the year the first Pokemon game was released?"

logger.debug(llm_chain.invoke(question))

"""
## Distributed Inference

vLLM supports distributed tensor-parallel inference and serving. 

To run multi-GPU inference with the LLM class, set the `tensor_parallel_size` argument to the number of GPUs you want to use. For example, to run inference on 4 GPUs
"""
logger.info("## Distributed Inference")


llm = VLLM(
    model="mosaicml/mpt-30b",
    tensor_parallel_size=4,
    trust_remote_code=True,  # mandatory for hf models
)

llm.invoke("What is the future of AI?")

"""
## Quantization

vLLM supports `awq` quantization. To enable it, pass `quantization` to `vllm_kwargs`.
"""
logger.info("## Quantization")

llm_q = VLLM(
    model="TheBloke/Llama-2-7b-Chat-AWQ",
    trust_remote_code=True,
    max_new_tokens=512,
    vllm_kwargs={"quantization": "awq"},
)

"""
## Ollama-Compatible Server

vLLM can be deployed as a server that mimics the Ollama API protocol. This allows vLLM to be used as a drop-in replacement for applications using Ollama API.

This server can be queried in the same format as Ollama API.

### Ollama-Compatible Completion
"""
logger.info("## Ollama-Compatible Server")


llm = VLLMOllama(
    ollama_ollama_api_base="http://localhost:8000/v1",
    model_name="tiiuae/falcon-7b",
    model_kwargs={"stop": ["."]},
)
logger.debug(llm.invoke("Rome is"))

"""
## LoRA adapter
LoRA adapters can be used with any vLLM model that implements `SupportsLoRA`.
"""
logger.info("## LoRA adapter")


llm = VLLM(
    model="meta-llama/Llama-3.2-3B-Instruct",
    max_new_tokens=300,
    top_k=1,
    top_p=0.90,
    temperature=0.1,
    vllm_kwargs={
        "gpu_memory_utilization": 0.5,
        "enable_lora": True,
        "max_model_len": 350,
    },
)
LoRA_ADAPTER_PATH = "path/to/adapter"
lora_adapter = LoRARequest("lora_adapter", 1, LoRA_ADAPTER_PATH)

logger.debug(
    llm.invoke("What are some popular Korean street foods?", lora_request=lora_adapter)
)

logger.info("\n\n[DONE]", bright=True)