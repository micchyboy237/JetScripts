from jet.logger import logger
from langchain_core.documents import Document
from langchain_pinecone import PineconeRerank
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
# Pinecone Rerank

> This notebook shows how to use **PineconeRerank** for two-stage vector retrieval reranking using Pinecone's hosted reranking API as demonstrated in `langchain_pinecone/libs/pinecone/rerank.py`.

## Setup
Install the `langchain-pinecone` package.
"""
logger.info("# Pinecone Rerank")

# %pip install -qU "langchain-pinecone"

"""
## Credentials
Set your Pinecone API key to use the reranking API.
"""
logger.info("## Credentials")

# from getpass import getpass

# os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY") or getpass(
    "Enter your Pinecone API key: "
)

"""
## Instantiation
Use `PineconeRerank` to rerank a list of documents by relevance to a query.
"""
logger.info("## Instantiation")


reranker = PineconeRerank(model="bge-reranker-v2-m3")

documents = [
    Document(page_content="Paris is the capital of France."),
    Document(page_content="Berlin is the capital of Germany."),
    Document(page_content="The Eiffel Tower is in Paris."),
]

query = "What is the capital of France?"
reranked_docs = reranker.compress_documents(documents, query)

for doc in reranked_docs:
    score = doc.metadata.get("relevance_score")
    logger.debug(f"Score: {score:.4f} | Content: {doc.page_content}")

"""
## Usage
### Reranking with Top-N
Specify `top_n` to limit the number of returned documents.
"""
logger.info("## Usage")

reranker_top1 = PineconeRerank(model="bge-reranker-v2-m3", top_n=1)
top1_docs = reranker_top1.compress_documents(documents, query)
logger.debug("Top-1 Result:")
for doc in top1_docs:
    logger.debug(f"Score: {doc.metadata['relevance_score']:.4f} | Content: {doc.page_content}")

"""
## Reranking with Custom Rank Fields
If your documents are dictionaries or have custom fields, use `rank_fields` to specify the field to rank on.
"""
logger.info("## Reranking with Custom Rank Fields")

docs_dict = [
    {
        "id": "doc1",
        "text": "Article about renewable energy.",
        "title": "Renewable Energy",
    },
    {"id": "doc2", "text": "Report on economic growth.", "title": "Economic Growth"},
    {
        "id": "doc3",
        "text": "News on climate policy changes.",
        "title": "Climate Policy",
    },
]

reranker_text = PineconeRerank(model="bge-reranker-v2-m3", rank_fields=["text"])
climate_docs = reranker_text.rerank(docs_dict, "Latest news on climate change.")

for res in climate_docs:
    logger.debug(f"ID: {res['id']} | Score: {res['score']:.4f}")

"""
We can rerank based on title field
"""
logger.info("We can rerank based on title field")

economic_docs = reranker_text.rerank(docs_dict, "Economic forecast.")

for res in economic_docs:
    logger.debug(
        f"ID: {res['id']} | Score: {res['score']:.4f} | Title: {res['document']['title']}"
    )

"""
## Reranking with Additional Parameters
You can pass model-specific parameters (e.g., `truncate`) directly to `.rerank()`.

How to handle inputs longer than those supported by the model. Accepted values: END or NONE.
END truncates the input sequence at the input token limit. NONE returns an error when the input exceeds the input token limit.
"""
logger.info("## Reranking with Additional Parameters")

docs_simple = [
    {"id": "docA", "text": "Quantum entanglement is a physical phenomenon..."},
    {"id": "docB", "text": "Classical mechanics describes motion..."},
]

reranked = reranker.rerank(
    documents=docs_simple,
    query="Explain the concept of quantum entanglement.",
    truncate="END",
)
for res in reranked:
    logger.debug(f"ID: {res['id']} | Score: {res['score']:.4f}")

"""
## Use within a chain

## API reference
- `PineconeRerank(model, top_n, rank_fields, return_documents)`
- `.rerank(documents, query, rank_fields=None, model=None, top_n=None, truncate="END")`
- `.compress_documents(documents, query)` (returns `Document` objects with `relevance_score` in metadata)
"""
logger.info("## Use within a chain")

logger.info("\n\n[DONE]", bright=True)