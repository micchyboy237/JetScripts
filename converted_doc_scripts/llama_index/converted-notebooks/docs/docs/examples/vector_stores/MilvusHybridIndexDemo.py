from FlagEmbedding import BGEM3FlagModel
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BGEM3SparseEmbeddingFunction
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction
from llama_index.vector_stores.milvus.utils import BaseSparseEmbeddingFunction
from typing import List
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

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/MilvusHybridIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Milvus Vector Store With Hybrid Search

Hybrid search leverages the strengths of both semantic retrieval and keyword matching to deliver more accurate and contextually relevant results. By combining the advantages of semantic search and keyword matching, hybrid search is particularly effective in complex information retrieval tasks.

This notebook demonstrates how to use Milvus for hybrid search in LlamaIndex RAG pipelines. We'll begin with the recommended default hybrid search (semantic + BM25) and then explore other alternative sparse embedding methods and customization of hybrid reranker.

## Prerequisites

**Install dependencies**

Before getting started, make sure you have the following dependencies installed:
"""
logger.info("# Milvus Vector Store With Hybrid Search")

# ! pip install llama-index-vector-stores-milvus
# ! pip install llama-index-embeddings-ollama
# ! pip install llama-index-llms-ollama

"""
> If you're using Google Colab, you may need to **restart the runtime** (Navigate to the "Runtime" menu at the top of the interface, and select "Restart session" from the dropdown menu.)

**Set up accounts**

This tutorial uses MLX for text embeddings and answer generation. You need to prepare the [MLX API key](https://platform.openai.com/api-keys).
"""
logger.info("This tutorial uses MLX for text embeddings and answer generation. You need to prepare the [MLX API key](https://platform.openai.com/api-keys).")


openai.api_key = "sk-"

"""
To use the Milvus vector store, specify your Milvus server `URI` (and optionally with the `TOKEN`). To start a Milvus server, you can set up a Milvus server by following the [Milvus installation guide](https://milvus.io/docs/install-overview.md) or simply trying [Zilliz Cloud](https://docs.zilliz.com/docs/register-with-zilliz-cloud) for free.

> Full-text search is currently supported in Milvus Standalone, Milvus Distributed, and Zilliz Cloud, but not yet in Milvus Lite (planned for future implementation). Reach out support@zilliz.com for more information.
"""
logger.info("To use the Milvus vector store, specify your Milvus server `URI` (and optionally with the `TOKEN`). To start a Milvus server, you can set up a Milvus server by following the [Milvus installation guide](https://milvus.io/docs/install-overview.md) or simply trying [Zilliz Cloud](https://docs.zilliz.com/docs/register-with-zilliz-cloud) for free.")

URI = "http://localhost:19530"

"""
**Load example data**

Run the following commands to download sample documents into the f"{GENERATED_DIR}/paul_graham" directory:
"""
logger.info("Run the following commands to download sample documents into the f"{GENERATED_DIR}/paul_graham" directory:")

# ! mkdir -p 'data/paul_graham/'
# ! wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
Then use `SimpleDirectoryReaderLoad` to load the essay "What I Worked On" by Paul Graham:
"""
logger.info("Then use `SimpleDirectoryReaderLoad` to load the essay "What I Worked On" by Paul Graham:")


documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

logger.debug("Example document:\n", documents[0])

"""
## Hybrid Search with BM25

This section shows how to perform a hybrid search using BM25. To get started, we will initialize the `MilvusVectorStore` and create an index for the example documents. The default configuration uses:

- Dense embeddings from the default embedding model (MLX's `text-embedding-ada-002`)
- BM25 for full-text search if enable_sparse is True
- RRFRanker with k=60 for combining results if hybrid search is enabled
"""
logger.info("## Hybrid Search with BM25")



vector_store = MilvusVectorStore(
    uri=URI,
    dim=1536,  # vector dimension depends on the embedding model
    enable_sparse=True,  # enable the default full-text search using BM25
    overwrite=True,  # drop the collection if it already exists
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
Here is more information about the arguments for configuring dense and sparse fields in the `MilvusVectorStore`:

**dense field**
- `enable_dense (bool)`: A boolean flag to enable or disable dense embedding. Defaults to True.
- `dim (int, optional)`: The dimension of the embedding vectors for the collection.
- `embedding_field (str, optional)`: The name of the dense embedding field for the collection, defaults to DEFAULT_EMBEDDING_KEY.
- `index_config (dict, optional)`: The configuration used for building the dense embedding index. Defaults to None.
- `search_config (dict, optional)`: The configuration used for searching the Milvus dense index. Note that this must be compatible with the index type specified by `index_config`. Defaults to None.
- `similarity_metric (str, optional)`: The similarity metric to use for dense embedding, currently supports IP, COSINE and L2.

**sparse field**
- `enable_sparse (bool)`: A boolean flag to enable or disable sparse embedding. Defaults to False.
- `sparse_embedding_field (str)`: The name of sparse embedding field, defaults to DEFAULT_SPARSE_EMBEDDING_KEY.
- `sparse_embedding_function (Union[BaseSparseEmbeddingFunction, BaseMilvusBuiltInFunction], optional)`: If enable_sparse is True, this object should be provided to convert text to a sparse embedding. If None, the default sparse embedding function (BM25BuiltInFunction) will be used, or use BGEM3SparseEmbedding given existing collection without built-in functions.
- `sparse_index_config (dict, optional)`: The configuration used to build the sparse embedding index. Defaults to None.

To enable hybrid search during the querying stage, set `vector_store_query_mode` to "hybrid". This will combine and rerank search results from both semantic search and full-text search. Let's test with a sample query: "What did the author learn at Viaweb?":
"""
logger.info("Here is more information about the arguments for configuring dense and sparse fields in the `MilvusVectorStore`:")


query_engine = index.as_query_engine(
    vector_store_query_mode="hybrid", similarity_top_k=5
)
response = query_engine.query("What did the author learn at Viaweb?")
logger.debug(textwrap.fill(str(response), 100))

"""
### Customize text analyzer

Analyzers play a vital role in full-text search by breaking sentences into tokens and performing lexical processing, such as stemming and stop-word removal. They are typically language-specific. For more details, refer to [Milvus Analyzer Guide](https://milvus.io/docs/analyzer-overview.md#Analyzer-Overview).

Milvus supports two types of analyzers: **Built-in Analyzers** and **Custom Analyzers**. By default, if `enable_sparse` is set to True, `MilvusVectorStore` utilizes the `BM25BuiltInFunction` with default configurations, employing the standard built-in analyzer that tokenizes text based on punctuation.

To use a different analyzer or customize the existing one, you can provide values to the `analyzer_params` argument when building the `BM25BuiltInFunction`. Then, set this function as the `sparse_embedding_function` in `MilvusVectorStore`.
"""
logger.info("### Customize text analyzer")


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

vector_store = MilvusVectorStore(
    uri=URI,
    dim=1536,
    enable_sparse=True,
    sparse_embedding_function=bm25_function,  # BM25 with custom analyzer
    overwrite=True,
)

"""
## Hybrid Search with Other Sparse Embedding

Besides combining semantic search with BM25, Milvus also supports hybrid search using a sparse embedding function such as [BGE-M3](https://arxiv.org/abs/2402.03216). The following example uses the built-in `BGEM3SparseEmbeddingFunction` to generate sparse embeddings.

First, we need to install the `FlagEmbedding` package:
"""
logger.info("## Hybrid Search with Other Sparse Embedding")

# ! pip install -q FlagEmbedding

"""
Then let's build the vector store and index using the default MLX model for densen embedding and the built-in BGE-M3 for sparse embedding:
"""
logger.info("Then let's build the vector store and index using the default MLX model for densen embedding and the built-in BGE-M3 for sparse embedding:")


vector_store = MilvusVectorStore(
    uri=URI,
    dim=1536,
    enable_sparse=True,
    sparse_embedding_function=BGEM3SparseEmbeddingFunction(),
    overwrite=True,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

"""
Now let's perform a hybrid search query with a sample question:
"""
logger.info("Now let's perform a hybrid search query with a sample question:")

query_engine = index.as_query_engine(
    vector_store_query_mode="hybrid", similarity_top_k=5
)
response = query_engine.query("What did the author learn at Viaweb??")
logger.debug(textwrap.fill(str(response), 100))

"""
### Customize Sparse Embedding Function

You can also customize the sparse embedding function as long as it inherits from `BaseSparseEmbeddingFunction`, including the following methods:

- `encode_queries`: This method converts texts into list of sparse embeddings for queries.
- `encode_documents`: This method converts text into list of sparse embeddings for documents.

The output of each method should follow the format of the sparse embedding, which is a list of dictionaries. Each dictionary should have a key (an integer) representing the dimension, and a corresponding value (a float) representing the embedding's magnitude in that dimension (e.g., {1: 0.5, 2: 0.3}).

For example, here's a custom sparse embedding function implementation using BGE-M3:
"""
logger.info("### Customize Sparse Embedding Function")



class ExampleEmbeddingFunction(BaseSparseEmbeddingFunction):
    def __init__(self):
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

    def encode_queries(self, queries: List[str]):
        outputs = self.model.encode(
            queries,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )["lexical_weights"]
        return [self._to_standard_dict(output) for output in outputs]

    def encode_documents(self, documents: List[str]):
        outputs = self.model.encode(
            documents,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )["lexical_weights"]
        return [self._to_standard_dict(output) for output in outputs]

    def _to_standard_dict(self, raw_output):
        result = {}
        for k in raw_output:
            result[int(k)] = raw_output[k]
        return result

"""
## Customize hybrid reranker

Milvus supports two types of [reranking strategies](https://milvus.io/docs/reranking.md): Reciprocal Rank Fusion (RRF) and Weighted Scoring. The default ranker in `MilvusVectorStore` hybrid search is RRF with k=60. To customize the hybrid ranker, modify the following parameters:

- `hybrid_ranker (str)`: Specifies the type of ranker used in hybrid search queries. Currently only supports ["RRFRanker", "WeightedRanker"]. Defaults to "RRFRanker".
- `hybrid_ranker_params (dict, optional)`: Configuration parameters for the hybrid ranker. The structure of this dictionary depends on the specific ranker being used:
    - For "RRFRanker", it should include:
        - "k" (int): A parameter used in Reciprocal Rank Fusion (RRF). This value is used to calculate the rank scores as part of the RRF algorithm, which combines multiple ranking strategies into a single score to improve search relevance. The default value is 60 if not specified.
    - For "WeightedRanker", it expects:
        - "weights" (list of float): A list of exactly two weights:
            1. The weight for the dense embedding component.
            2. The weight for the sparse embedding component.
            These weights are used to balance the significance of the dense and sparse components of the embeddings in the hybrid retrieval process. The default weights are [1.0, 1.0] if not specified.
"""
logger.info("## Customize hybrid reranker")

vector_store = MilvusVectorStore(
    uri=URI,
    dim=1536,
    overwrite=False,  # Use the existing collection created in the previous example
    enable_sparse=True,
    hybrid_ranker="WeightedRanker",
    hybrid_ranker_params={"weights": [1.0, 0.5]},
)
index = VectorStoreIndex.from_vector_store(vector_store)
query_engine = index.as_query_engine(
    vector_store_query_mode="hybrid", similarity_top_k=5
)
response = query_engine.query("What did the author learn at Viaweb?")
logger.debug(textwrap.fill(str(response), 100))

logger.info("\n\n[DONE]", bright=True)