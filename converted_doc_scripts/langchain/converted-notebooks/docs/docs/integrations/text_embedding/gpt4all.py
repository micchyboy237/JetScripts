from jet.logger import logger
from langchain_community.embeddings import GPT4AllEmbeddings
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
# GPT4All

[GPT4All](https://gpt4all.io/index.html) is a free-to-use, locally running, privacy-aware chatbot. There is no GPU or internet required. It features popular models and its own models such as GPT4All Falcon, Wizard, etc.

This notebook explains how to use [GPT4All embeddings](https://docs.gpt4all.io/gpt4all_python_embedding.html#gpt4all.gpt4all.Embed4All) with LangChain.

## Install GPT4All's Python Bindings
"""
logger.info("# GPT4All")

# %pip install --upgrade --quiet  gpt4all > /dev/null

"""
Note: you may need to restart the kernel to use updated packages.
"""
logger.info("Note: you may need to restart the kernel to use updated packages.")


gpt4all_embd = GPT4AllEmbeddings()

text = "This is a test document."

"""
## Embed the Textual Data
"""
logger.info("## Embed the Textual Data")

query_result = gpt4all_embd.embed_query(text)

"""
With embed_documents you can embed multiple pieces of text. You can also map these embeddings with [Nomic's Atlas](https://docs.nomic.ai/index.html) to see a visual representation of your data.
"""
logger.info("With embed_documents you can embed multiple pieces of text. You can also map these embeddings with [Nomic's Atlas](https://docs.nomic.ai/index.html) to see a visual representation of your data.")

doc_result = gpt4all_embd.embed_documents([text])

logger.info("\n\n[DONE]", bright=True)