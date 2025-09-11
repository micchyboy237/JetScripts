from jet.logger import logger
from langchain_community.llms import Predibase
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
# Predibase

Learn how to use LangChain with models on Predibase.

## Setup

- Create a [Predibase](https://predibase.com/) account and [API key](https://docs.predibase.com/sdk-guide/intro).
- Install the Predibase Python client with `pip install predibase`
- Use your API key to authenticate

### LLM

Predibase integrates with LangChain by implementing LLM module. You can see a short example below or a full notebook under LLM > Integrations > Predibase.
"""
logger.info("# Predibase")

os.environ["PREDIBASE_API_TOKEN"] = "{PREDIBASE_API_TOKEN}"


model = Predibase(
    model="mistral-7b",
    predibase_api_key=os.environ.get("PREDIBASE_API_TOKEN"),
    predibase_sdk_version=None,  # optional parameter (defaults to the latest Predibase SDK version if omitted)
    """
    Optionally use `model_kwargs` to set new default "generate()" settings.  For example:
    {
        "api_token": os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        "max_new_tokens": 5,  # default is 256
    }
    """
    **model_kwargs,
)

"""
Optionally use `kwargs` to dynamically overwrite "generate()" settings.  For example:
{
    "temperature": 0.5,  # default is the value in model_kwargs or 0.1 (initialization default)
    "max_new_tokens": 1024,  # default is the value in model_kwargs or 256 (initialization default)
}
"""
response = model.invoke("Can you recommend me a nice dry wine?", **kwargs)
logger.debug(response)

"""
Predibase also supports Predibase-hosted and HuggingFace-hosted adapters that are fine-tuned on the base model given by the `model` argument:
"""
logger.info("Predibase also supports Predibase-hosted and HuggingFace-hosted adapters that are fine-tuned on the base model given by the `model` argument:")

os.environ["PREDIBASE_API_TOKEN"] = "{PREDIBASE_API_TOKEN}"


model = Predibase(
    model="mistral-7b",
    predibase_api_key=os.environ.get("PREDIBASE_API_TOKEN"),
    predibase_sdk_version=None,  # optional parameter (defaults to the latest Predibase SDK version if omitted)
    adapter_id="e2e_nlg",
    adapter_version=1,
    """
    Optionally use `model_kwargs` to set new default "generate()" settings.  For example:
    {
        "api_token": os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        "max_new_tokens": 5,  # default is 256
    }
    """
    **model_kwargs,
)

"""
Optionally use `kwargs` to dynamically overwrite "generate()" settings.  For example:
{
    "temperature": 0.5,  # default is the value in model_kwargs or 0.1 (initialization default)
    "max_new_tokens": 1024,  # default is the value in model_kwargs or 256 (initialization default)
}
"""
response = model.invoke("Can you recommend me a nice dry wine?", **kwargs)
logger.debug(response)

"""
Predibase also supports adapters that are fine-tuned on the base model given by the `model` argument:
"""
logger.info("Predibase also supports adapters that are fine-tuned on the base model given by the `model` argument:")

os.environ["PREDIBASE_API_TOKEN"] = "{PREDIBASE_API_TOKEN}"


model = Predibase(
    model="mistral-7b",
    predibase_api_key=os.environ.get("PREDIBASE_API_TOKEN"),
    predibase_sdk_version=None,  # optional parameter (defaults to the latest Predibase SDK version if omitted)
    adapter_id="predibase/e2e_nlg",
    """
    Optionally use `model_kwargs` to set new default "generate()" settings.  For example:
    {
        "api_token": os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        "max_new_tokens": 5,  # default is 256
    }
    """
    **model_kwargs,
)

"""
Optionally use `kwargs` to dynamically overwrite "generate()" settings.  For example:
{
    "temperature": 0.5,  # default is the value in model_kwargs or 0.1 (initialization default)
    "max_new_tokens": 1024,  # default is the value in model_kwargs or 256 (initialization default)
}
"""
response = model.invoke("Can you recommend me a nice dry wine?", **kwargs)
logger.debug(response)

logger.info("\n\n[DONE]", bright=True)