import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.mymagic import MyMagicAI
import asyncio
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
# MyMagic AI LLM

## Introduction
This notebook demonstrates how to use MyMagicAI for batch inference on massive data stored in cloud buckets. The only enpoints implemented are `complete` and `acomplete` which can work on many use cases including Completion, Summariation and Extraction.
To use this notebook, you need an API key (Personal Access Token) from MyMagicAI and data stored in cloud buckets.
Sign up by clicking Get Started at [MyMagicAI's website](https://mymagic.ai/) to get your API key.

## Setup
To set up your bucket and grant MyMagic API a secure access to your cloud storage, please visit [MyMagic docs](https://docs.mymagic.ai/) for reference.
If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# MyMagic AI LLM")

# %pip install llama-index-llms-mymagic

# !pip install llama-index


llm = MyMagicAI(
    api_key="your-api-key",
    storage_provider="s3",  # s3, gcs
    bucket_name="your-bucket-name",
    session="your-session-name",  # files should be located in this folder on which batch inference will be run
    role_arn="your-role-arn",
    system_prompt="your-system-prompt",
    region="your-bucket-region",
    return_output=False,  # Whether you want MyMagic API to return the output json
    input_json_file=None,  # name of the input file (stored on the bucket)
    list_inputs=None,  # Option to provide inputs as a list in case of small batch
    structured_output=None,  # json schema of the output
)

"""
Note: if return_output is set True above, max_tokens should be set to at least 100
"""
logger.info("Note: if return_output is set True above, max_tokens should be set to at least 100")

resp = llm.complete(
    question="your-question",
    model="chhoose-model",  # currently we support mistral7b, llama7b, mixtral8x7b, codellama70b, llama70b, more to come...
    max_tokens=5,  # number of tokens to generate, default is 10
)

logger.debug(resp)

"""
## Asynchronous Requests by using `acomplete` endpoint
For asynchronous operations, use the following approach.
"""
logger.info("## Asynchronous Requests by using `acomplete` endpoint")


async def main():
    async def async_func_3():
        response = llm.complete(
            question="your-question",
            model="choose-model",  # supported models constantly updated and are listed at docs.mymagic.ai
            max_tokens=5,  # number of tokens to generate, default is 10
        )
        return response
    response = asyncio.run(async_func_3())
    logger.success(format_json(response))

    logger.debug("Async completion response:", response)

async def run_async_code_ba09313d():
    await main()
    return 
 = asyncio.run(run_async_code_ba09313d())
logger.success(format_json())

logger.info("\n\n[DONE]", bright=True)