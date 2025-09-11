from jet.logger import logger
from langchain_community.embeddings import LlamafileEmbeddings
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
# llamafile

Let's load the [llamafile](https://github.com/Mozilla-Ocho/llamafile) Embeddings class.

## Setup

First, the are 3 setup steps:

1. Download a llamafile. In this notebook, we use `TinyLlama-1.1B-Chat-v1.0.Q5_K_M` but there are many others available on [HuggingFace](https://huggingface.co/models?other=llamafile).
2. Make the llamafile executable.
3. Start the llamafile in server mode.

You can run the following bash script to do all this:
"""
logger.info("# llamafile")

# %%bash

wget -nv -nc https://huggingface.co/jartine/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile

chmod +x TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile

./TinyLlama-1.1B-Chat-v1.0.Q5_K_M.llamafile --server --nobrowser --embedding > tinyllama.log 2>&1 &
pid=$!
echo "${pid}" > .llamafile_pid  # write the process pid to a file so we can terminate the server later

"""
## Embedding texts using LlamafileEmbeddings

Now, we can use the `LlamafileEmbeddings` class to interact with the llamafile server that's currently serving our TinyLlama model at http://localhost:8080.
"""
logger.info("## Embedding texts using LlamafileEmbeddings")


embedder = LlamafileEmbeddings()

text = "This is a test document."

"""
To generate embeddings, you can either query an invidivual text, or you can query a list of texts.
"""
logger.info("To generate embeddings, you can either query an invidivual text, or you can query a list of texts.")

query_result = embedder.embed_query(text)
query_result[:5]

doc_result = embedder.embed_documents([text])
doc_result[0][:5]

# %%bash
kill $(cat .llamafile_pid)
rm .llamafile_pid

logger.info("\n\n[DONE]", bright=True)