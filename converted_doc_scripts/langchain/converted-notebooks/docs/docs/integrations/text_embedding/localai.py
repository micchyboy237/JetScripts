from jet.logger import logger
from langchain_community.embeddings import LocalAIEmbeddings
from langchain_localai import LocalAIEmbeddings
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
# LocalAI

:::info

`langchain-localai` is a 3rd party integration package for LocalAI. It provides a simple way to use LocalAI services in Langchain.

The source code is available on [Github](https://github.com/mkhludnev/langchain-localai)

:::

Let's load the LocalAI Embedding class. In order to use the LocalAI Embedding class, you need to have the LocalAI service hosted somewhere and configure the embedding models. See the documentation at https://localai.io/basics/getting_started/index.html and https://localai.io/features/embeddings/index.html.
"""
logger.info("# LocalAI")

# %pip install -U langchain-localai


embeddings = LocalAIEmbeddings(
    ollama_api_base="http://localhost:8080", model="embedding-model-name"
)

text = "This is a test document."

query_result = embeddings.embed_query(text)

doc_result = embeddings.embed_documents([text])

"""
Let's load the LocalAI Embedding class with first generation models (e.g. text-search-ada-doc-001/text-search-ada-query-001). Note: These are not recommended models - see [here](https://platform.ollama.com/docs/guides/embeddings/what-are-embeddings)
"""
logger.info("Let's load the LocalAI Embedding class with first generation models (e.g. text-search-ada-doc-001/text-search-ada-query-001). Note: These are not recommended models - see [here](https://platform.ollama.com/docs/guides/embeddings/what-are-embeddings)")


embeddings = LocalAIEmbeddings(
    ollama_api_base="http://localhost:8080", model="embedding-model-name"
)

text = "This is a test document."

query_result = embeddings.embed_query(text)

doc_result = embeddings.embed_documents([text])


os.environ["OPENAI_PROXY"] = "http://proxy.yourcompany.com:8080"

logger.info("\n\n[DONE]", bright=True)