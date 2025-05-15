from dotenv import load_dotenv
from jet.llm.ollama.base import Ollama
from jet.llm.ollama.base import OllamaEmbedding
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import BaseNode, TransformComponent
from llama_index.legacy.retrievers.bm25_retriever import BM25Retriever
from llama_index.vector_stores.faiss import FaissVectorStore
from typing import List
import faiss
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join(script_dir, "generated", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

"""
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/fusion_retrieval_with_llamaindex.ipynb)

# Fusion Retrieval in Document Search

## Overview

This code implements a Fusion Retrieval system that combines vector-based similarity search with keyword-based BM25 retrieval. The approach aims to leverage the strengths of both methods to improve the overall quality and relevance of document retrieval.

## Motivation

Traditional retrieval methods often rely on either semantic understanding (vector-based) or keyword matching (BM25). Each approach has its strengths and weaknesses. Fusion retrieval aims to combine these methods to create a more robust and accurate retrieval system that can handle a wider range of queries effectively.

## Key Components

1. PDF processing and text chunking
2. Vector store creation using FAISS and Ollama embeddings
3. BM25 index creation for keyword-based retrieval
4. Fusioning BM25 and vector search results for better retrieval

## Method Details

### Document Preprocessing

1. The PDF is loaded and split into chunks using SentenceSplitter.
2. Chunks are cleaned by replacing 't' with spaces and newline cleaning (likely addressing a specific formatting issue).

### Vector Store Creation

1. Ollama embeddings are used to create vector representations of the text chunks.
2. A FAISS vector store is created from these embeddings for efficient similarity search.

### BM25 Index Creation

1. A BM25 index is created from the same text chunks used for the vector store.
2. This allows for keyword-based retrieval alongside the vector-based method.

### Query Fusion Retrieval

After creation of both indexes Query Fusion Retrieval combines them to enable a hybrid retrieval

## Benefits of this Approach

1. Improved Retrieval Quality: By combining semantic and keyword-based search, the system can capture both conceptual similarity and exact keyword matches.
2. Flexibility: The `retriever_weights` parameter allows for adjusting the balance between vector and keyword search based on specific use cases or query types.
3. Robustness: The combined approach can handle a wider range of queries effectively, mitigating weaknesses of individual methods.
4. Customizability: The system can be easily adapted to use different vector stores or keyword-based retrieval methods.

## Conclusion

Fusion retrieval represents a powerful approach to document search that combines the strengths of semantic understanding and keyword matching. By leveraging both vector-based and BM25 retrieval methods, it offers a more comprehensive and flexible solution for information retrieval tasks. This approach has potential applications in various fields where both conceptual similarity and keyword relevance are important, such as academic research, legal document search, or general-purpose search engines.

# Package Installation and Imports

The cell below installs all necessary packages required to run this notebook.
"""
logger.info("# Fusion Retrieval in Document Search")

# !pip install faiss-cpu llama-index python-dotenv


load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

EMBED_DIMENSION = 512
Settings.llm = Ollama(model="llama3.2", request_timeout=300.0,
                      context_window=4096, temperature=0.1)
Settings.embed_model = OllamaEmbedding(model_name="mxbai-embed-large")

"""
### Read Docs
"""
logger.info("### Read Docs")

os.makedirs('data', exist_ok=True)

# !wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/N7/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf

path = f"{GENERATED_DIR}/"
reader = SimpleDirectoryReader(input_dir=path, required_exts=['.pdf'])
documents = reader.load_data()
logger.debug(documents[0])

"""
### Create Vector Store
"""
logger.info("### Create Vector Store")

fais_index = faiss.IndexFlatL2(EMBED_DIMENSION)
vector_store = FaissVectorStore(faiss_index=fais_index)

"""
### Text Cleaner Transformation
"""
logger.info("### Text Cleaner Transformation")


class TextCleaner(TransformComponent):
    """
    Transformation to be used within the ingestion pipeline.
    Cleans clutters from texts.
    """

    def __call__(self, nodes, **kwargs) -> List[BaseNode]:

        for node in nodes:
            node.text = node.text.replace(
                '\t', ' ')  # Replace tabs with spaces
            # Replace paragprah seperator with spacaes
            node.text = node.text.replace(' \n', ' ')

        return nodes


"""
### Ingestion Pipeline
"""
logger.info("### Ingestion Pipeline")

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(),
        TextCleaner()
    ],
    vector_store=vector_store,
    documents=documents
)

nodes = pipeline.run()

"""
## Retrievers

### BM25 Retriever
"""
logger.info("## Retrievers")

bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=2,
)

"""
### Vector Retriever
"""
logger.info("### Vector Retriever")

index = VectorStoreIndex(nodes)
vector_retriever = index.as_retriever(similarity_top_k=2)

"""
### Fusing Both Retrievers
"""
logger.info("### Fusing Both Retrievers")

retriever = QueryFusionRetriever(
    retrievers=[
        vector_retriever,
        bm25_retriever
    ],
    retriever_weights=[
        0.6,  # vector retriever weight
        0.4  # BM25 retriever weight
    ],
    num_queries=1,
    mode='dist_based_score',
    use_async=False
)

"""
About parameters

1. `num_queries`:  Query Fusion Retriever not only combines retrievers but also can genereate multiple questions from a given query. This parameter controls how many total queries will be passed to the retrievers. Therefore setting it to 1 disables query generation and the final retriever only uses the initial query.
2. `mode`: There are 4 options for this parameter. 
   - **reciprocal_rerank**: Applies reciporical ranking. (Since there is no normalization, this method is not suitable for this kind of application. Beacuse different retrirevers will return score scales)
   - **relative_score**: Applies MinMax based on the min and max scores among all the nodes. Then scaled to be between 0 and 1. Finally scores are weighted by the relative retrievers based on `retriever_weights`.  
      ```math
      min\_score = min(scores)
      \\ max\_score = max(scores)
      ```
   - **dist_based_score**:  Only difference from `relative_score` is the MinMax sclaing is based on mean and std of the scores. Scaling and weighting is the same.
      ```math
       min\_score = mean\_score - 3 * std\_dev
      \\ max\_score = mean\_score + 3 * std\_dev
      ```
   - **simple**: This method is simply takes the max score of each chunk.

### Use Case example
"""
logger.info("### Use Case example")

query = "What are the impacts of climate change on the environment?"

response = retriever.retrieve(query)

"""
### Print Final Retrieved Nodes with Scores
"""
logger.info("### Print Final Retrieved Nodes with Scores")

for node in response:
    logger.debug(f"Node Score: {node.score:.2}")
    logger.debug(f"Node Content: {node.text}")
    logger.debug("-"*100)

logger.info("\n\n[DONE]", bright=True)
