from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from llama_index.core import DocumentSummaryIndex
from llama_index.core import SimpleDirectoryReader, get_response_synthesizer
from llama_index.core import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core.indices.document_summary import (
DocumentSummaryIndexEmbeddingRetriever,
)
from llama_index.core.indices.document_summary import (
DocumentSummaryIndexLLMRetriever,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from pathlib import Path
import logging
import openai
import os
import requests
import shutil
import sys


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/index_structs/doc_summary/DocSummary.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Document Summary Index

This demo showcases the document summary index, over Wikipedia articles on different cities.

The document summary index will extract a summary from each document and store that summary, as well as all nodes corresponding to the document.

Retrieval can be performed through the LLM or embeddings (which is a TODO). We first select the relevant documents to the query based on their summaries. All retrieved nodes corresponding to the selected documents are retrieved.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Document Summary Index")

# %pip install llama-index-llms-ollama

# !pip install llama-index


# os.environ["OPENAI_API_KEY"] = "sk-..."
# openai.api_key = os.environ["OPENAI_API_KEY"]


logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# import nest_asyncio

# nest_asyncio.apply()


"""
### Load Datasets

Load Wikipedia pages on different cities
"""
logger.info("### Load Datasets")

wiki_titles = ["Toronto", "Seattle", "Chicago", "Boston", "Houston"]



for title in wiki_titles:
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
        },
    ).json()
    page = next(iter(response["query"]["pages"].values()))
    wiki_text = page["extract"]

    data_path = Path("data")
    if not data_path.exists():
        Path.mkdir(data_path)

    with open(data_path / f"{title}.txt", "w") as fp:
        fp.write(wiki_text)

city_docs = []
for wiki_title in wiki_titles:
    docs = SimpleDirectoryReader(
        input_files=[ff"{GENERATED_DIR}/{wiki_title}.txt"]
    ).load_data()
    docs[0].doc_id = wiki_title
    city_docs.extend(docs)

"""
### Build Document Summary Index

We show two ways of building the index:
- default mode of building the document summary index
- customizing the summary query
"""
logger.info("### Build Document Summary Index")

chatgpt = MLX(temperature=0, model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats")
splitter = SentenceSplitter(chunk_size=1024)

response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize", use_async=True
)
doc_summary_index = DocumentSummaryIndex.from_documents(
    city_docs,
    llm=chatgpt,
    transformations=[splitter],
    response_synthesizer=response_synthesizer,
    show_progress=True,
)

doc_summary_index.get_document_summary("Boston")

doc_summary_index.storage_context.persist("index")


storage_context = StorageContext.from_defaults(persist_dir="index")
doc_summary_index = load_index_from_storage(storage_context)

"""
### Perform Retrieval from Document Summary Index

We show how to execute queries at a high-level. We also show how to perform retrieval at a lower-level so that you can view the parameters that are in place. We show both LLM-based retrieval and embedding-based retrieval using the document summaries.

#### High-level Querying

Note: this uses the default, embedding-based form of retrieval
"""
logger.info("### Perform Retrieval from Document Summary Index")

query_engine = doc_summary_index.as_query_engine(
    response_mode="tree_summarize", use_async=True
)

response = query_engine.query("What are the sports teams in Toronto?")

logger.debug(response)

"""
#### LLM-based Retrieval
"""
logger.info("#### LLM-based Retrieval")


retriever = DocumentSummaryIndexLLMRetriever(
    doc_summary_index,
)

retrieved_nodes = retriever.retrieve("What are the sports teams in Toronto?")

logger.debug(len(retrieved_nodes))

logger.debug(retrieved_nodes[0].score)
logger.debug(retrieved_nodes[0].node.get_text())


response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

response = query_engine.query("What are the sports teams in Toronto?")
logger.debug(response)

"""
#### Embedding-based Retrieval
"""
logger.info("#### Embedding-based Retrieval")


retriever = DocumentSummaryIndexEmbeddingRetriever(
    doc_summary_index,
)

retrieved_nodes = retriever.retrieve("What are the sports teams in Toronto?")

len(retrieved_nodes)

logger.debug(retrieved_nodes[0].node.get_text())


response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

response = query_engine.query("What are the sports teams in Toronto?")
logger.debug(response)

logger.info("\n\n[DONE]", bright=True)