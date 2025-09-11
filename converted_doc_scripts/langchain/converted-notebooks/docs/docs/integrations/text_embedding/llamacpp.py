from jet.logger import logger
from langchain_community.embeddings import LlamaCppEmbeddings
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
# Llama.cpp

>[llama.cpp python](https://github.com/abetlen/llama-cpp-python) library is a simple Python bindings for `@ggerganov`
>[llama.cpp](https://github.com/ggerganov/llama.cpp).
>
>This package provides:
>
> - Low-level access to C API via ctypes interface.
> - High-level Python API for text completion
>   - `Ollama`-like API
>   - `LangChain` compatibility
>   - `LlamaIndex` compatibility
> - Ollama compatible web server
>   - Local Copilot replacement
>   - Function Calling support
>   - Vision API support
>   - Multiple Models
"""
logger.info("# Llama.cpp")

# %pip install --upgrade --quiet  llama-cpp-python


llama = LlamaCppEmbeddings(model_path="/path/to/model/ggml-model-q4_0.bin")

text = "This is a test document."

query_result = llama.embed_query(text)

doc_result = llama.embed_documents([text])

logger.info("\n\n[DONE]", bright=True)