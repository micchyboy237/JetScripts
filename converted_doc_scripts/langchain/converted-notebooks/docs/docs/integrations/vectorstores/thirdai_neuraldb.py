from jet.logger import logger
from langchain_community.vectorstores import NeuralDBVectorStore
from thirdai import neural_db as ndb
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
# ThirdAI NeuralDB

>[NeuralDB](https://www.thirdai.com/neuraldb-enterprise/) is a CPU-friendly and fine-tunable vector store developed by [ThirdAI](https://www.thirdai.com/).

## Initialization

There are two initialization methods:
- From Scratch: Basic model
- From Checkpoint: Load a model that was previously saved

For all of the following initialization methods, the `thirdai_key` parameter can be omitted if the `THIRDAI_KEY` environment variable is set.

ThirdAI API keys can be obtained at https://www.thirdai.com/try-bolt/

You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration
"""
logger.info("# ThirdAI NeuralDB")


vectorstore = NeuralDBVectorStore.from_scratch(thirdai_key="your-thirdai-key")

vectorstore = NeuralDBVectorStore.from_checkpoint(
    checkpoint="/path/to/checkpoint.ndb",
    thirdai_key="your-thirdai-key",
)

"""
## Inserting document sources
"""
logger.info("## Inserting document sources")

vectorstore.insert(
    sources=["/path/to/doc.pdf", "/path/to/doc.docx", "/path/to/doc.csv"],
    train=True,
    fast_mode=True,
)


vectorstore.insert(
    sources=[
        ndb.PDF(
            "/path/to/doc.pdf",
            version="v2",
            chunk_size=100,
            metadata={"published": 2022},
        ),
        ndb.Unstructured("/path/to/deck.pptx"),
    ]
)

"""
## Similarity search

To query the vectorstore, you can use the standard LangChain vectorstore method `similarity_search`, which returns a list of LangChain Document objects. Each document object represents a chunk of text from the indexed files. For example, it may contain a paragraph from one of the indexed PDF files. In addition to the text, the document's metadata field contains information such as the document's ID, the source of this document (which file it came from), and the score of the document.
"""
logger.info("## Similarity search")

documents = vectorstore.similarity_search("query", k=10)

"""
## Fine tuning

NeuralDBVectorStore can be fine-tuned to user behavior and domain-specific knowledge. It can be fine-tuned in two ways:
1. Association: the vectorstore associates a source phrase with a target phrase. When the vectorstore sees the source phrase, it will also consider results that are relevant to the target phrase.
2. Upvoting: the vectorstore upweights the score of a document for a specific query. This is useful when you want to fine-tune the vectorstore to user behavior. For example, if a user searches "how is a car manufactured" and likes the returned document with id 52, then we can upvote the document with id 52 for the query "how is a car manufactured".
"""
logger.info("## Fine tuning")

vectorstore.associate(source="source phrase", target="target phrase")
vectorstore.associate_batch(
    [
        ("source phrase 1", "target phrase 1"),
        ("source phrase 2", "target phrase 2"),
    ]
)

vectorstore.upvote(query="how is a car manufactured", document_id=52)
vectorstore.upvote_batch(
    [
        ("query 1", 52),
        ("query 2", 20),
    ]
)

logger.info("\n\n[DONE]", bright=True)