from jet.models.config import MODELS_CACHE_DIR
from jet.logger import logger
from langchain_huggingface import HuggingFaceEmbeddings
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
# Sentence Transformers on Hugging Face

>[Hugging Face sentence-transformers](https://huggingface.co/sentence-transformers) is a Python framework for state-of-the-art sentence, text and image embeddings.
>You can use these embedding models from the `HuggingFaceEmbeddings` class.

:::caution

Running sentence-transformers locally can be affected by your operating system and other global factors. It is recommended for experienced users only.

:::

## Setup

You'll need to install the `langchain_huggingface` package as a dependency:
"""
logger.info("# Sentence Transformers on Hugging Face")

# %pip install -qU langchain-huggingface

"""
## Usage
"""
logger.info("## Usage")


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

text = "This is a test document."
query_result = embeddings.embed_query(text)

logger.debug(str(query_result)[:100] + "...")

doc_result = embeddings.embed_documents([text, "This is not a test document."])
logger.debug(str(doc_result)[:100] + "...")

"""
## Troubleshooting

If you are having issues with the `accelerate` package not being found or failing to import, installing/upgrading it may help:
"""
logger.info("## Troubleshooting")

# %pip install -qU accelerate

logger.info("\n\n[DONE]", bright=True)