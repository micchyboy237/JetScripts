from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction
import openai
import os
import shutil
import textwrap


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/MilvusFullTextSearchDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Milvus Vector Store with Full-Text Search

**Full-text search** uses exact keyword matching, often leveraging algorithms like BM25 to rank documents by relevance. In **Retrieval-Augmented Generation (RAG)** systems, this method retrieves pertinent text to enhance AI-generated responses.

Meanwhile, **semantic search** interprets contextual meaning to provide broader results. Combining both approaches creates a **hybrid search** that improves information retrieval—especially in cases where a single method falls short.

With [Milvus 2.5](https://milvus.io/blog/introduce-milvus-2-5-full-text-search-powerful-metadata-filtering-and-more.md)'s Sparse-BM25 approach, raw text is automatically converted into sparse vectors. This eliminates the need for manual sparse embedding generation and enables a hybrid search strategy that balances semantic understanding with keyword relevance.

In this tutorial, you'll learn how to use LlamaIndex and Milvus to build a RAG system using full-text search and hybrid search. We'll start by implementing full-text search alone and then enhance it by integrating semantic search for more comprehensive results.

> Before proceeding with this tutorial, ensure you are familiar with [full-text search](https://milvus.io/docs/full-text-search.md#Full-Text-Search) and the [basics of using Milvus in LlamaIndex](https://milvus.io/docs/integrate_with_llamaindex.md).

## Prerequisites

**Install dependencies**

Before getting started, make sure you have the following dependencies installed:
"""
logger.info("# Milvus Vector Store with Full-Text Search")

# %pip install llama-index-vector-stores-milvus
# %pip install llama-index-embeddings-huggingface
# %pip install llama-index-llms-ollama

"""
> If you're using Google Colab, you may need to **restart the runtime** (Navigate to the "Runtime" menu at the top of the interface, and select "Restart session" from the dropdown menu.)

**Set up accounts**

This tutorial uses OllamaFunctionCalling for text embeddings and answer generation. You need to prepare the [OllamaFunctionCalling API key](https://platform.openai.com/api-keys).
"""
logger.info("This tutorial uses OllamaFunctionCalling for text embeddings and answer generation. You need to prepare the [OllamaFunctionCalling API key](https://platform.openai.com/api-keys).")


openai.api_key = "sk-"

"""
To use the Milvus vector store, specify your Milvus server `URI` (and optionally with the `TOKEN`). To start a Milvus server, you can set up a Milvus server by following the [Milvus installation guide](https://milvus.io/docs/install-overview.md) or simply trying [Zilliz Cloud](https://docs.zilliz.com/docs/register-with-zilliz-cloud) for free.

> Full-text search is currently supported in Milvus Standalone, Milvus Distributed, and Zilliz Cloud, but not yet in Milvus Lite (planned for future implementation). Reach out support@zilliz.com for more information.
"""
logger.info("To use the Milvus vector store, specify your Milvus server `URI` (and optionally with the `TOKEN`). To start a Milvus server, you can set up a Milvus server by following the [Milvus installation guide](https://milvus.io/docs/install-overview.md) or simply trying [Zilliz Cloud](https://docs.zilliz.com/docs/register-with-zilliz-cloud) for free.")

URI = "http://localhost:19530"

"""
**Download example data**

Run the following commands to download sample documents into the "data/paul_graham" directory:
"""
logger.info("Run the following commands to download sample documents into the "data/paul_graham" directory:")

# %mkdir -p 'data/paul_graham/'
# %wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
## RAG with Full-Text Search

Integrating full-text search into a RAG system balances semantic search with precise and predictable keyword-based retrieval. You can also choose to only use full text search though it's recommended to combine full text search with semantic search for better search results. Here for demonstration purpose we will show full text search alone and hybrid search.

To get started, use `SimpleDirectoryReaderLoad` to load the essay "What I Worked On" by Paul Graham:
"""
logger.info("## RAG with Full-Text Search")


documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

logger.debug("Example document:\n", documents[0])

"""
### Full-Text Search with BM25

LlamaIndex's `MilvusVectorStore` supports full-text search, enabling efficient keyword-based retrieval. By using a built-in function as the `sparse_embedding_function`, it applies BM25 scoring to rank search results.

In this section, we’ll demonstrate how to implement a RAG system using BM25 for full-text search.
"""
logger.info("### Full-Text Search with BM25")


Settings.embed_model = None

vector_store = MilvusVectorStore(
    uri=URI,
    enable_dense=False,
    enable_sparse=True,  # Only enable sparse to demo full text search
    sparse_embedding_function=BM25BuiltInFunction(),
    overwrite=True,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
The above code inserts example documents into Milvus and builds an index to enable BM25 ranking for full-text search. It disables dense embedding and utilizes `BM25BuiltInFunction` with default parameters.

You can specify the input and output fields in the `BM25BuiltInFunction` parameters:

- `input_field_names (str)`: The input text field (default: "text"). It indicates which text field the BM25 algorithm applied to. Change this if using your own collection with a different text field name.
- `output_field_names (str)`: The field where outputs of this BM25 function are stored (default: "sparse_embedding").

Once the vector store is set up, you can perform full-text search queries using Milvus with query mode "sparse" or "text_search":
"""
logger.info("The above code inserts example documents into Milvus and builds an index to enable BM25 ranking for full-text search. It disables dense embedding and utilizes `BM25BuiltInFunction` with default parameters.")


query_engine = index.as_query_engine(
    vector_store_query_mode="sparse", similarity_top_k=5
)
answer = query_engine.query("What did the author learn at Viaweb?")
logger.debug(textwrap.fill(str(answer), 100))

"""
#### Customize text analyzer

Analyzers play a vital role in full-text search by breaking sentences into tokens and performing lexical processing, such as stemming and stop-word removal. They are typically language-specific. For more details, refer to [Milvus Analyzer Guide](https://milvus.io/docs/analyzer-overview.md#Analyzer-Overview).

Milvus supports two types of analyzers: **Built-in Analyzers** and **Custom Analyzers**. By default, the `BM25BuiltInFunction` uses the standard built-in analyzer, which tokenizes text based on punctuation.

To use a different analyzer or customize the existing one, you can pass value to the `analyzer_params` argument:
"""
logger.info("#### Customize text analyzer")

bm25_function = BM25BuiltInFunction(
    analyzer_params={
        "tokenizer": "standard",
        "filter": [
            "lowercase",  # Built-in filter
            {"type": "length", "max": 40},  # Custom cap size of a single token
            {"type": "stop", "stop_words": ["of", "to"]},  # Custom stopwords
        ],
    },
    enable_match=True,
)

"""
### Hybrid Search with Reranker

A hybrid search system combines semantic search and full-text search, optimizing retrieval performance in a RAG system.

The following example uses OllamaFunctionCalling embedding for semantic search and BM25 for full-text search:
"""
logger.info("### Hybrid Search with Reranker")

vector_store = MilvusVectorStore(
    uri=URI,
    dim=1536,
    enable_sparse=True,
    sparse_embedding_function=BM25BuiltInFunction(),
    overwrite=True,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model="default",  # "default" will use OllamaFunctionCalling embedding
)

"""
**How it works**

This approach stores documents in a Milvus collection with both vector fields:

- `embedding`: Dense embeddings generated by OllamaFunctionCalling embedding model for semantic search.
- `sparse_embedding`: Sparse embeddings computed using BM25BuiltInFunction for full-text search.

In addition, we have applied a reranking strategy using "RRFRanker" with its default parameters. To customize reranker, you are able to configure `hybrid_ranker` and `hybrid_ranker_params` following the [Milvus Reranking Guide](https://milvus.io/docs/reranking.md).

Now, let's test the RAG system with a sample query:
"""
logger.info("This approach stores documents in a Milvus collection with both vector fields:")

query_engine = index.as_query_engine(
    vector_store_query_mode="hybrid", similarity_top_k=5
)
answer = query_engine.query("What did the author learn at Viaweb?")
logger.debug(textwrap.fill(str(answer), 100))

"""
This hybrid approach ensures more accurate, context-aware responses in a RAG system by leveraging both semantic and keyword-based retrieval.
"""
logger.info("This hybrid approach ensures more accurate, context-aware responses in a RAG system by leveraging both semantic and keyword-based retrieval.")

logger.info("\n\n[DONE]", bright=True)