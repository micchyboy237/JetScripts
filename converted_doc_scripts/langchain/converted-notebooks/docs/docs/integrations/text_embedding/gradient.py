from jet.logger import logger
from langchain_community.embeddings import GradientEmbeddings
import numpy as np
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
# Gradient

`Gradient` allows to create `Embeddings` as well fine tune and get completions on LLMs with a simple web API.

This notebook goes over how to use Langchain with Embeddings of [Gradient](https://gradient.ai/).

## Imports
"""
logger.info("# Gradient")


"""
## Set the Environment API Key
Make sure to get your API key from Gradient AI. You are given $10 in free credits to test and fine-tune different models.
"""
logger.info("## Set the Environment API Key")

# from getpass import getpass

if not os.environ.get("GRADIENT_ACCESS_TOKEN", None):
#     os.environ["GRADIENT_ACCESS_TOKEN"] = getpass("gradient.ai access token:")
if not os.environ.get("GRADIENT_WORKSPACE_ID", None):
#     os.environ["GRADIENT_WORKSPACE_ID"] = getpass("gradient.ai workspace id:")

"""
Optional: Validate your environment variables ```GRADIENT_ACCESS_TOKEN``` and ```GRADIENT_WORKSPACE_ID``` to get currently deployed models. Using the `gradientai` Python package.
"""
logger.info("Optional: Validate your environment variables ```GRADIENT_ACCESS_TOKEN``` and ```GRADIENT_WORKSPACE_ID``` to get currently deployed models. Using the `gradientai` Python package.")

# %pip install --upgrade --quiet  gradientai

"""
## Create the Gradient instance
"""
logger.info("## Create the Gradient instance")

documents = [
    "Pizza is a dish.",
    "Paris is the capital of France",
    "numpy is a lib for linear algebra",
]
query = "Where is Paris?"

embeddings = GradientEmbeddings(model="bge-large")

documents_embedded = embeddings.embed_documents(documents)
query_result = embeddings.embed_query(query)


scores = np.array(documents_embedded) @ np.array(query_result).T
dict(zip(documents, scores))

logger.info("\n\n[DONE]", bright=True)