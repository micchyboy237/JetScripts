import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.alephalpha import AlephAlpha
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/alephalpha.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Aleph Alpha

Aleph Alpha is a powerful language model that can generate human-like text. Aleph Alpha is capable of generating text in multiple languages and styles, and can be fine-tuned to generate text in specific domains.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Aleph Alpha")

# %pip install llama-index-llms-alephalpha

# !pip install llama-index

"""
#### Set your Aleph Alpha token
"""
logger.info("#### Set your Aleph Alpha token")


os.environ["AA_TOKEN"] = "your_token_here"

"""
#### Call `complete` with a prompt
"""
logger.info("#### Call `complete` with a prompt")


llm = AlephAlpha(model="luminous-base-control")

resp = llm.complete("Paul Graham is ")

logger.debug(resp)

"""
#### Additional Response Details
To access detailed response information such as log probabilities, ensure your AlephAlpha instance is initialized with the `log_probs` parameter. The `logprobs` attribute of the `CompletionResponse` will contain this data. Other details like the model version and raw completion text can be accessed directly if they're part of the response or via `additional_kwargs`.
"""
logger.info("#### Additional Response Details")


llm = AlephAlpha(model="luminous-base-control", log_probs=0)

resp = llm.complete("Paul Graham is ")

if resp.logprobs is not None:
    logger.debug("\nLog Probabilities:")
    for lp_list in resp.logprobs:
        for lp in lp_list:
            logger.debug(f"Token: {lp.token}, LogProb: {lp.logprob}")

if "model_version" in resp.additional_kwargs:
    logger.debug("\nModel Version:")
    logger.debug(resp.additional_kwargs["model_version"])

if "raw_completion" in resp.additional_kwargs:
    logger.debug("\nRaw Completion:")
    logger.debug(resp.additional_kwargs["raw_completion"])

"""
## Async
"""
logger.info("## Async")


llm = AlephAlpha(model="luminous-base-control")
async def run_async_code_c3ecd675():
    async def run_async_code_a989c387():
        resp = llm.complete("Paul Graham is ")
        return resp
    resp = asyncio.run(run_async_code_a989c387())
    logger.success(format_json(resp))
    return resp
resp = asyncio.run(run_async_code_c3ecd675())
logger.success(format_json(resp))

logger.debug(resp)

logger.info("\n\n[DONE]", bright=True)