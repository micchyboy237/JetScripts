from jet.logger import logger
from langchain_community.retrievers import NeuralDBRetriever
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
# **NeuralDB**
NeuralDB is a CPU-friendly and fine-tunable retrieval engine developed by ThirdAI.

### **Initialization**
There are two initialization methods:
- From Scratch: Basic model
- From Checkpoint: Load a model that was previously saved

For all of the following initialization methods, the `thirdai_key` parameter can be ommitted if the `THIRDAI_KEY` environment variable is set.

ThirdAI API keys can be obtained at https://www.thirdai.com/try-bolt/
"""
logger.info("# **NeuralDB**")


retriever = NeuralDBRetriever.from_scratch(thirdai_key="your-thirdai-key")

retriever = NeuralDBRetriever.from_checkpoint(
    checkpoint="/path/to/checkpoint.ndb",
    thirdai_key="your-thirdai-key",
)

"""
### **Inserting document sources**
"""
logger.info("### **Inserting document sources**")

retriever.insert(
    sources=["/path/to/doc.pdf", "/path/to/doc.docx", "/path/to/doc.csv"],
    train=True,
    fast_mode=True,
)


retriever.insert(
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
### **Retrieving documents**
To query the retriever, you can use the standard LangChain retriever method `get_relevant_documents`, which returns a list of LangChain Document objects. Each document object represents a chunk of text from the indexed files. For example, it may contain a paragraph from one of the indexed PDF files. In addition to the text, the document's metadata field contains information such as the document's ID, the source of this document (which file it came from), and the score of the document.
"""
logger.info("### **Retrieving documents**")

documents = retriever.invoke("query", top_k=10)

"""
### **Fine tuning**
NeuralDBRetriever can be fine-tuned to user behavior and domain-specific knowledge. It can be fine-tuned in two ways:
1. Association: the retriever associates a source phrase with a target phrase. When the retriever sees the source phrase, it will also consider results that are relevant to the target phrase.
2. Upvoting: the retriever upweights the score of a document for a specific query. This is useful when you want to fine-tune the retriever to user behavior. For example, if a user searches "how is a car manufactured" and likes the returned document with id 52, then we can upvote the document with id 52 for the query "how is a car manufactured".
"""
logger.info("### **Fine tuning**")

retriever.associate(source="source phrase", target="target phrase")
retriever.associate_batch(
    [
        ("source phrase 1", "target phrase 1"),
        ("source phrase 2", "target phrase 2"),
    ]
)

retriever.upvote(query="how is a car manufactured", document_id=52)
retriever.upvote_batch(
    [
        ("query 1", 52),
        ("query 2", 20),
    ]
)

logger.info("\n\n[DONE]", bright=True)