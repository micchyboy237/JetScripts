from jet.llm.mlx.base import MLX
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
import os
import qdrant_client
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# RAG Bootcamp ◦ February 2024 ◦ Vector Institute
"""
logger.info("# RAG Bootcamp ◦ February 2024 ◦ Vector Institute")



"""
![Title Image](https://d3ddy8balm3goa.cloudfront.net/rag-bootcamp-vector/title.excalidraw.svg)

![Title Image](https://d3ddy8balm3goa.cloudfront.net/rag-bootcamp-vector/framework.excalidraw.svg)

#### Notebook Setup & Dependency Installation
"""
logger.info("#### Notebook Setup & Dependency Installation")

# %pip install llama-index llama-index-vector-stores-qdrant -q

# import nest_asyncio

# nest_asyncio.apply()

# !mkdir data
# !wget "https://arxiv.org/pdf/2402.09353.pdf" -O "./data/dorav1.pdf"

"""
## Motivation

![Motivation Image](https://d3ddy8balm3goa.cloudfront.net/rag-bootcamp-vector/motivation.excalidraw.svg)
"""
logger.info("## Motivation")


llm = MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats")
response = llm.complete("What is DoRA?")

logger.debug(response.text)

"""
## Basic RAG in 3 Steps

![Divider Image](https://d3ddy8balm3goa.cloudfront.net/rag-bootcamp-vector/subheading.excalidraw.svg)


1. Build external knowledge (i.e., updated data sources)
2. Retrieve
3. Augment and Generate

## 1. Build External Knowledge

![Divider Image](https://d3ddy8balm3goa.cloudfront.net/rag-bootcamp-vector/step1.excalidraw.svg)
"""
logger.info("## Basic RAG in 3 Steps")

"""Load the data.

With llama-index, before any transformations are applied,
data is loaded in the `Document` abstraction, which is
a container that holds the text of the document.
"""


loader = SimpleDirectoryReader(input_dir="./data")
documents = loader.load_data()



"""Chunk, Encode, and Store into a Vector Store.

To streamline the process, we can make use of the IngestionPipeline
class that will apply your specified transformations to the
Document's.
"""


client = qdrant_client.QdrantClient(location=":memory:")
vector_store = QdrantVectorStore(client=client, collection_name="test_store")

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(),
        MLXEmbedding(),
    ],
    vector_store=vector_store,
)
_nodes = pipeline.run(documents=documents, num_workers=4)



"""Create a llama-index... wait for it... Index.

After uploading your encoded documents into your vector
store of choice, you can connect to it with a VectorStoreIndex
which then gives you access to all of the llama-index functionality.
"""


index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

"""
## 2. Retrieve Against A Query

![Step2 Image](https://d3ddy8balm3goa.cloudfront.net/rag-bootcamp-vector/step2.excalidraw.svg)
"""
logger.info("## 2. Retrieve Against A Query")

"""Retrieve relevant documents against a query.

With our Index ready, we can now query it to
retrieve the most relevant document chunks.
"""

retriever = index.as_retriever(similarity_top_k=2)
retrieved_nodes = retriever.retrieve("What is DoRA?")

"""
## 3. Generate Final Response

![Step3 Image](https://d3ddy8balm3goa.cloudfront.net/rag-bootcamp-vector/step3.excalidraw.svg)
"""
logger.info("## 3. Generate Final Response")

"""Context-Augemented Generation.

With our Index ready, we can create a QueryEngine
that handles the retrieval and context augmentation
in order to get the final response.
"""

query_engine = index.as_query_engine()

logger.debug(
    query_engine.get_prompts()[
        "response_synthesizer:text_qa_template"
    ].default_template.template
)

response = query_engine.query("What is DoRA?")
logger.debug(response)

"""
## In Summary

- LLMs as powerful as they are, don't perform too well with knowledge-intensive tasks (domain-specific, updated data, long-tail)
- Context augmentation has been shown (in a few studies) to outperform LLMs without augmentation
- In this notebook, we showed one such example that follows that pattern.

# LlamaIndex Has More To Offer

- Data infrastructure that enables production-grade, advanced RAG systems
- Agentic solutions
- Newly released: `llama-index-networks`
- Enterprise offerings (alpha):
    - LlamaParse (proprietary complex PDF parser) and
    - LlamaCloud

### Useful links

[website](https://www.llamaindex.ai/) ◦ [llamahub](https://llamahub.ai) ◦ [github](https://github.com/run-llama/llama_index) ◦ [medium](https://medium.com/@llama_index) ◦ [rag-bootcamp-poster](https://d3ddy8balm3goa.cloudfront.net/rag-bootcamp-vector/final_poster.excalidraw.svg)
"""
logger.info("## In Summary")

logger.info("\n\n[DONE]", bright=True)