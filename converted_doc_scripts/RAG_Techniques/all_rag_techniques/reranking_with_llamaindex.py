from dotenv import load_dotenv
from jet.llm.ollama.base import Ollama
from jet.llm.ollama.base import OllamaEmbedding
from jet.logger import CustomLogger
from llama_index.core import Document
from llama_index.core import QueryBundle
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank, LLMRerank
from llama_index.core.readers import SimpleDirectoryReader
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
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/reranking_with_llamaindex.ipynb)

# Reranking Methods in RAG Systems

## Overview
Reranking is a crucial step in Retrieval-Augmented Generation (RAG) systems that aims to improve the relevance and quality of retrieved documents. It involves reassessing and reordering initially retrieved documents to ensure that the most pertinent information is prioritized for subsequent processing or presentation.

## Motivation
The primary motivation for reranking in RAG systems is to overcome limitations of initial retrieval methods, which often rely on simpler similarity metrics. Reranking allows for more sophisticated relevance assessment, taking into account nuanced relationships between queries and documents that might be missed by traditional retrieval techniques. This process aims to enhance the overall performance of RAG systems by ensuring that the most relevant information is used in the generation phase.

## Key Components
Reranking systems typically include the following components:

1. Initial Retriever: Often a vector store using embedding-based similarity search.
2. Reranking Model: This can be either:
   - A Large Language Model (LLM) for scoring relevance
   - A Cross-Encoder model specifically trained for relevance assessment
3. Scoring Mechanism: A method to assign relevance scores to documents
4. Sorting and Selection Logic: To reorder documents based on new scores

## Method Details
The reranking process generally follows these steps:

1. Initial Retrieval: Fetch an initial set of potentially relevant documents.
2. Pair Creation: Form query-document pairs for each retrieved document.
3. Scoring: 
   - LLM Method: Use prompts to ask the LLM to rate document relevance.
   - Cross-Encoder Method: Feed query-document pairs directly into the model.
4. Score Interpretation: Parse and normalize the relevance scores.
5. Reordering: Sort documents based on their new relevance scores.
6. Selection: Choose the top K documents from the reordered list.

## Benefits of this Approach
Reranking offers several advantages:

1. Improved Relevance: By using more sophisticated models, reranking can capture subtle relevance factors.
2. Flexibility: Different reranking methods can be applied based on specific needs and resources.
3. Enhanced Context Quality: Providing more relevant documents to the RAG system improves the quality of generated responses.
4. Reduced Noise: Reranking helps filter out less relevant information, focusing on the most pertinent content.

## Conclusion
Reranking is a powerful technique in RAG systems that significantly enhances the quality of retrieved information. Whether using LLM-based scoring or specialized Cross-Encoder models, reranking allows for more nuanced and accurate assessment of document relevance. This improved relevance translates directly to better performance in downstream tasks, making reranking an essential component in advanced RAG implementations.

The choice between LLM-based and Cross-Encoder reranking methods depends on factors such as required accuracy, available computational resources, and specific application needs. Both approaches offer substantial improvements over basic retrieval methods and contribute to the overall effectiveness of RAG systems.

<div style="text-align: center;">

<img src="../images/reranking-visualization.svg" alt="rerank llm" style="width:100%; height:auto;">
</div>

<div style="text-align: center;">

<img src="../images/reranking_comparison.svg" alt="rerank llm" style="width:100%; height:auto;">
</div>

# Package Installation and Imports

The cell below installs all necessary packages required to run this notebook.
"""
logger.info("# Reranking Methods in RAG Systems")

# !pip install faiss-cpu llama-index python-dotenv


load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

EMBED_DIMENSION = 512
Settings.llm = Ollama(
    model="llama3.2", request_timeout=300.0, context_window=4096)
Settings.embed_model = OllamaEmbedding(model_name="mxbai-embed-large")

"""
### Read docs
"""
logger.info("### Read docs")

os.makedirs('data', exist_ok=True)

# !wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/N7/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf

path = f"{GENERATED_DIR}/"
reader = SimpleDirectoryReader(input_dir=path, required_exts=['.pdf'])
documents = reader.load_data()

"""
### Create a vector store
"""
logger.info("### Create a vector store")

fais_index = faiss.IndexFlatL2(EMBED_DIMENSION)
vector_store = FaissVectorStore(faiss_index=fais_index)

"""
## Ingestion Pipeline
"""
logger.info("## Ingestion Pipeline")

base_pipeline = IngestionPipeline(
    transformations=[SentenceSplitter()],
    vector_store=vector_store,
    documents=documents
)

nodes = base_pipeline.run()

"""
## Querying

### Method 1: LLM based reranking the retrieved documents

<div style="text-align: center;">

<img src="../images/rerank_llm.svg" alt="rerank llm" style="width:40%; height:auto;">
</div>
"""
logger.info("## Querying")

index = VectorStoreIndex(nodes)

query_engine_w_llm_rerank = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[
        LLMRerank(
            top_n=5
        )
    ],
)

resp = query_engine_w_llm_rerank.query(
    "What are the impacts of climate change on biodiversity?")
logger.debug(resp)

"""
#### Example that demonstrates why we should use reranking
"""
logger.info("#### Example that demonstrates why we should use reranking")

chunks = [
    "The capital of France is great.",
    "The capital of France is huge.",
    "The capital of France is beautiful.",
    """Have you ever visited Paris? It is a beautiful city where you can eat delicious food and see the Eiffel Tower. I really enjoyed all the cities in france, but its capital with the Eiffel Tower is my favorite city.""",
    "I really enjoyed my trip to Paris, France. The city is beautiful and the food is delicious. I would love to visit again. Such a great capital city."
]
docs = [Document(page_content=sentence) for sentence in chunks]


def compare_rag_techniques(query: str, docs: List[Document] = docs) -> None:
    docs = [Document(text=sentence) for sentence in chunks]
    index = VectorStoreIndex.from_documents(docs)

    logger.debug("Comparison of Retrieval Techniques")
    logger.debug("==================================")
    logger.debug(f"Query: {query}\n")

    logger.debug("Baseline Retrieval Result:")
    baseline_docs = index.as_retriever(similarity_top_k=5).retrieve(query)
    # Get only the first two retrieved docs
    for i, doc in enumerate(baseline_docs[:2]):
        logger.debug(f"\nDocument {i+1}:")
        logger.debug(doc.text)

    logger.debug("\nAdvanced Retrieval Result:")
    reranker = LLMRerank(
        top_n=2,
    )
    advanced_docs = reranker.postprocess_nodes(
        baseline_docs,
        QueryBundle(query)
    )
    for i, doc in enumerate(advanced_docs):
        logger.debug(f"\nDocument {i+1}:")
        logger.debug(doc.text)


query = "what is the capital of france?"
compare_rag_techniques(query, docs)

"""
### Method 2: Cross Encoder models

<div style="text-align: center;">

<img src="../images/rerank_cross_encoder.svg" alt="rerank cross encoder" style="width:40%; height:auto;">
</div>

LlamaIndex has builtin support for [SBERT](https://www.sbert.net/index.html) models that can be used directly as node postprocessor.
"""
logger.info("### Method 2: Cross Encoder models")

query_engine_w_cross_encoder = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[
        SentenceTransformerRerank(
            model='cross-encoder/ms-marco-MiniLM-L-6-v2',
            top_n=5
        )
    ],
)

resp = query_engine_w_cross_encoder.query(
    "What are the impacts of climate change on biodiversity?")
logger.debug(resp)

logger.info("\n\n[DONE]", bright=True)
