from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ipex_llm import IpexLLMEmbedding
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
# Local Embeddings with IPEX-LLM on Intel GPU

> [IPEX-LLM](https://github.com/intel-analytics/ipex-llm/) is a PyTorch library for running LLM on Intel CPU and GPU (e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max) with very low latency.

This example goes over how to use LlamaIndex to conduct embedding tasks with `ipex-llm` optimizations on Intel GPU. This would be helpful in applications such as RAG, document QA, etc.

> **Note**
>
> You could refer to [here](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/embeddings/llama-index-embeddings-ipex-llm/examples) for full examples of `IpexLLMEmbedding`. Please note that for running on Intel GPU, please specify `-d 'xpu'` or `-d 'xpu:<device_id>'` in command argument when running the examples.

## Install Prerequisites
To benefit from IPEX-LLM on Intel GPUs, there are several prerequisite steps for tools installation and environment preparation.

If you are a Windows user, visit the [Install IPEX-LLM on Windows with Intel GPU Guide](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/install_windows_gpu.html), and follow [**Install Prerequisites**](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/install_windows_gpu.html#install-prerequisites) to update GPU driver (optional) and install Conda.

If you are a Linux user, visit the [Install IPEX-LLM on Linux with Intel GPU](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/install_linux_gpu.html), and follow [**Install Prerequisites**](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/install_linux_gpu.html#install-prerequisites) to install GPU driver, Intel® oneAPI Base Toolkit 2024.0, and Conda.

## Install `llama-index-embeddings-ipex-llm`

After the prerequisites installation, you should have created a conda environment with all prerequisites installed, activate your conda environment and install `llama-index-embeddings-ipex-llm` as follows:

```bash
conda activate <your-conda-env-name>

pip install llama-index-embeddings-ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```
This step will also install `ipex-llm` and its dependencies.

> **Note**
>
> You can also use `https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/` as the `extra-indel-url`.


## Runtime Configuration

For optimal performance, it is recommended to set several environment variables based on your device:

### For Windows Users with Intel Core Ultra integrated GPU

In Anaconda Prompt:

```
set SYCL_CACHE_PERSISTENT=1
set BIGDL_LLM_XMX_DISABLED=1
```

### For Linux Users with Intel Arc A-Series GPU

```bash
# Configure oneAPI environment variables. Required step for APT or offline installed oneAPI.
# Skip this step for PIP-installed oneAPI since the environment has already been configured in LD_LIBRARY_PATH.
source /opt/intel/oneapi/setvars.sh

# Recommended Environment Variables for optimal performance
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_CACHE_PERSISTENT=1
```

> **Note**
>
> For the first time that each model runs on Intel iGPU/Intel Arc A300-Series or Pro A60, it may take several minutes to compile.
>
> For other GPU type, please refer to [here](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html#runtime-configuration) for Windows users, and  [here](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html#id5) for Linux users.

## `IpexLLMEmbedding`

Setting `device="xpu"` when initializing `IpexLLMEmbedding` will put the embedding model on Intel GPU and benefit from IPEX-LLM optimizations:

```python

embedding_model = IpexLLMEmbedding(
    model_name="BAAI/bge-large-en-v1.5", device="xpu"
)
```

> Please note that `IpexLLMEmbedding` currently only provides optimization for Hugging Face Bge models.
>
> If you have multiple Intel GPUs available, you could set `device="xpu:<device_id>"`, in which `device_id` is counted from 0. `device="xpu"` is equal to `device="xpu:0"` by default.

You could then conduct the embedding tasks as normal:

```python
sentence = "IPEX-LLM is a PyTorch library for running LLM on Intel CPU and GPU (e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max) with very low latency."
query = "What is IPEX-LLM?"

text_embedding = embedding_model.get_text_embedding(sentence)
logger.debug(f"embedding[:10]: {text_embedding[:10]}")

text_embeddings = embedding_model.get_text_embedding_batch([sentence, query])
logger.debug(f"text_embeddings[0][:10]: {text_embeddings[0][:10]}")
logger.debug(f"text_embeddings[1][:10]: {text_embeddings[1][:10]}")

query_embedding = embedding_model.get_query_embedding(query)
logger.debug(f"query_embedding[:10]: {query_embedding[:10]}")
```
"""
logger.info("# Local Embeddings with IPEX-LLM on Intel GPU")

logger.info("\n\n[DONE]", bright=True)