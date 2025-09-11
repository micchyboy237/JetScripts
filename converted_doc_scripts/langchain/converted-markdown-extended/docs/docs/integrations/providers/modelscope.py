from jet.logger import logger
from langchain_modelscope import ModelScopeChatEndpoint
from langchain_modelscope import ModelScopeEmbeddings
from langchain_modelscope import ModelScopeLLM
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
# ModelScope

>[ModelScope](https://www.modelscope.cn/home) is a big repository of the models and datasets.

This page covers how to use the modelscope ecosystem within LangChain.
It is broken into two parts: installation and setup, and then references to specific modelscope wrappers.

## Installation
"""
logger.info("# ModelScope")

pip install -U langchain-modelscope-integration

"""
Head to [ModelScope](https://modelscope.cn/) to sign up to ModelScope and generate an [SDK token](https://modelscope.cn/my/myaccesstoken). Once you've done this set the `MODELSCOPE_SDK_TOKEN` environment variable:
"""
logger.info("Head to [ModelScope](https://modelscope.cn/) to sign up to ModelScope and generate an [SDK token](https://modelscope.cn/my/myaccesstoken). Once you've done this set the `MODELSCOPE_SDK_TOKEN` environment variable:")

export MODELSCOPE_SDK_TOKEN=<your_sdk_token>

"""
## Chat Models

`ModelScopeChatEndpoint` class exposes chat models from ModelScope. See available models [here](https://www.modelscope.cn/docs/model-service/API-Inference/intro).
"""
logger.info("## Chat Models")


llm = ModelScopeChatEndpoint(model="Qwen/Qwen2.5-Coder-32B-Instruct")
llm.invoke("Sing a ballad of LangChain.")

"""
## Embeddings

`ModelScopeEmbeddings` class exposes embeddings from ModelScope.
"""
logger.info("## Embeddings")


embeddings = ModelScopeEmbeddings(model_id="damo/nlp_corom_sentence-embedding_english-base")
embeddings.embed_query("What is the meaning of life?")

"""
## LLMs
`ModelScopeLLM` class exposes LLMs from ModelScope.
"""
logger.info("## LLMs")


llm = ModelScopeLLM(model="Qwen/Qwen2.5-Coder-32B-Instruct")
llm.invoke("The meaning of life is")

logger.info("\n\n[DONE]", bright=True)