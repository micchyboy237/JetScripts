import asyncio
from jet.transformers.formatters import format_json
from IPython.display import display, Markdown
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.settings import Settings
from llama_index.core.vector_stores import VectorStoreQueryResult
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client import models
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import Any, List, Tuple
import os
import shutil
import torch


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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/qdrant_hybrid.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Qdrant Hybrid Search

Qdrant supports hybrid search by combining search results from `sparse` and `dense` vectors.

`dense` vectors are the ones you have probably already been using -- embedding models from MLX, BGE, SentenceTransformers, etc. are typically `dense` embedding models. They create a numerical representation of a piece of text, represented as a long list of numbers. These `dense` vectors can capture rich semantics across the entire piece of text.

`sparse` vectors are slightly different. They use a specialized approach or model (TF-IDF, BM25, SPLADE, etc.) for generating vectors. These vectors are typically mostly zeros, making them `sparse` vectors. These `sparse` vectors are great at capturing specific keywords and similar small details.

This notebook walks through setting up and customizing hybrid search with Qdrant and `"prithvida/Splade_PP_en_v1"` variants from Huggingface.

## Setup

First, we setup our env and load our data.
"""
logger.info("# Qdrant Hybrid Search")

# %pip install -U llama-index llama-index-vector-stores-qdrant fastembed


# os.environ["OPENAI_API_KEY"] = "sk-..."

# !mkdir -p 'data/'
# !wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O f"{GENERATED_DIR}/llama2.pdf"


documents = SimpleDirectoryReader("./data/").load_data()

"""
## Indexing Data

Now, we can index our data. 

Hybrid search with Qdrant must be enabled from the beginning -- we can simply set `enable_hybrid=True`.

This will run sparse vector generation locally using the `"prithvida/Splade_PP_en_v1"` using fastembed, in addition to generating dense vectors with MLX.
"""
logger.info("## Indexing Data")


client = QdrantClient(host="localhost", port=6333)
aclient = AsyncQdrantClient(host="localhost", port=6333)

vector_store = QdrantVectorStore(
    "llama2_paper",
    client=client,
    aclient=aclient,
    enable_hybrid=True,
    fastembed_sparse_model="Qdrant/bm25",
    batch_size=20,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
Settings.chunk_size = 512

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)

"""
## Hybrid Queries

When querying with hybrid mode, we can set `similarity_top_k` and `sparse_top_k` separately.

`sparse_top_k` represents how many nodes will be retrieved from each dense and sparse query. For example, if `sparse_top_k=5` is set, that means I will retrieve 5 nodes using sparse vectors and 5 nodes using dense vectors.

`similarity_top_k` controls the final number of returned nodes. In the above setting, we end up with 10 nodes. A fusion algorithm is applied to rank and order the nodes from different vector spaces ([relative score fusion](https://weaviate.io/blog/hybrid-search-fusion-algorithms#relative-score-fusion) in this case). `similarity_top_k=2` means the top two nodes after fusion are returned.
"""
logger.info("## Hybrid Queries")

query_engine = index.as_query_engine(
    similarity_top_k=2, sparse_top_k=12, vector_store_query_mode="hybrid"
)


response = query_engine.query(
    "How was Llama2 specifically trained differently from Llama1?"
)

display(Markdown(str(response)))

logger.debug(len(response.source_nodes))

"""
Lets compare to not using hybrid search at all!
"""
logger.info("Lets compare to not using hybrid search at all!")


query_engine = index.as_query_engine(
    similarity_top_k=2,
)

response = query_engine.query(
    "How was Llama2 specifically trained differently from Llama1?"
)
display(Markdown(str(response)))

"""
### Async Support

And of course, async queries are also supported (note that in-memory Qdrant data is not shared between async and sync clients!)
"""
logger.info("### Async Support")

# import nest_asyncio

# nest_asyncio.apply()



vector_store = QdrantVectorStore(
    collection_name="llama2_paper",
    client=client,
    aclient=aclient,
    enable_hybrid=True,
    fastembed_sparse_model="Qdrant/bm25",
    batch_size=20,
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
Settings.chunk_size = 512

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    use_async=True,
)

query_engine = index.as_query_engine(similarity_top_k=2, sparse_top_k=10)

async def async_func_28():
    response = query_engine.query(
        "What baseline models are measured against in the paper?"
    )
    return response
response = asyncio.run(async_func_28())
logger.success(format_json(response))

"""
## [Advanced] Customizing Hybrid Search with Qdrant

In this section, we walk through various settings that can be used to fully customize the hybrid search experience

### Customizing Sparse Vector Generation

Sparse vector generation can be done using a single model, or sometimes distinct separate models for queries and documents. Here we use two -- `"naver/efficient-splade-VI-BT-large-doc"` and `"naver/efficient-splade-VI-BT-large-query"`

Below is the sample code for generating the sparse vectors and how you can set the functionality in the constructor. You can use this and customize as needed.
"""
logger.info("## [Advanced] Customizing Hybrid Search with Qdrant")


doc_tokenizer = AutoTokenizer.from_pretrained(
    "naver/efficient-splade-VI-BT-large-doc"
)
doc_model = AutoModelForMaskedLM.from_pretrained(
    "naver/efficient-splade-VI-BT-large-doc"
)

query_tokenizer = AutoTokenizer.from_pretrained(
    "naver/efficient-splade-VI-BT-large-query"
)
query_model = AutoModelForMaskedLM.from_pretrained(
    "naver/efficient-splade-VI-BT-large-query"
)


def sparse_doc_vectors(
    texts: List[str],
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Computes vectors from logits and attention mask using ReLU, log, and max operations.
    """
    tokens = doc_tokenizer(
        texts, truncation=True, padding=True, return_tensors="pt"
    )
    if torch.cuda.is_available():
        tokens = tokens.to("cuda")

    output = doc_model(**tokens)
    logits, attention_mask = output.logits, tokens.attention_mask
    relu_log = torch.log(1 + torch.relu(logits))
    weighted_log = relu_log * attention_mask.unsqueeze(-1)
    tvecs, _ = torch.max(weighted_log, dim=1)

    indices = []
    vecs = []
    for batch in tvecs:
        indices.append(batch.nonzero(as_tuple=True)[0].tolist())
        vecs.append(batch[indices[-1]].tolist())

    return indices, vecs


def sparse_query_vectors(
    texts: List[str],
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Computes vectors from logits and attention mask using ReLU, log, and max operations.
    """
    tokens = query_tokenizer(
        texts, truncation=True, padding=True, return_tensors="pt"
    )
    if torch.cuda.is_available():
        tokens = tokens.to("cuda")

    output = query_model(**tokens)
    logits, attention_mask = output.logits, tokens.attention_mask
    relu_log = torch.log(1 + torch.relu(logits))
    weighted_log = relu_log * attention_mask.unsqueeze(-1)
    tvecs, _ = torch.max(weighted_log, dim=1)

    indices = []
    vecs = []
    for batch in tvecs:
        indices.append(batch.nonzero(as_tuple=True)[0].tolist())
        vecs.append(batch[indices[-1]].tolist())

    return indices, vecs

vector_store = QdrantVectorStore(
    "llama2_paper",
    client=client,
    enable_hybrid=True,
    sparse_doc_fn=sparse_doc_vectors,
    sparse_query_fn=sparse_query_vectors,
)

"""
### Customizing `hybrid_fusion_fn()`

By default, when running hbyrid queries with Qdrant, Relative Score Fusion is used to combine the nodes retrieved from both sparse and dense queries. 

You can customize this function to be any other method (plain deduplication, Reciprocal Rank Fusion, etc.).

Below is the default code for our relative score fusion approach and how you can pass it into the constructor.
"""
logger.info("### Customizing `hybrid_fusion_fn()`")



def relative_score_fusion(
    dense_result: VectorStoreQueryResult,
    sparse_result: VectorStoreQueryResult,
    alpha: float = 0.5,  # passed in from the query engine
    top_k: int = 2,  # passed in from the query engine i.e. similarity_top_k
) -> VectorStoreQueryResult:
    """
    Fuse dense and sparse results using relative score fusion.
    """
    assert dense_result.nodes is not None
    assert dense_result.similarities is not None
    assert sparse_result.nodes is not None
    assert sparse_result.similarities is not None

    sparse_result_tuples = list(
        zip(sparse_result.similarities, sparse_result.nodes)
    )
    sparse_result_tuples.sort(key=lambda x: x[0], reverse=True)

    dense_result_tuples = list(
        zip(dense_result.similarities, dense_result.nodes)
    )
    dense_result_tuples.sort(key=lambda x: x[0], reverse=True)

    all_nodes_dict = {x.node_id: x for x in dense_result.nodes}
    for node in sparse_result.nodes:
        if node.node_id not in all_nodes_dict:
            all_nodes_dict[node.node_id] = node

    sparse_similarities = [x[0] for x in sparse_result_tuples]
    max_sparse_sim = max(sparse_similarities)
    min_sparse_sim = min(sparse_similarities)
    sparse_similarities = [
        (x - min_sparse_sim) / (max_sparse_sim - min_sparse_sim)
        for x in sparse_similarities
    ]
    sparse_per_node = {
        sparse_result_tuples[i][1].node_id: x
        for i, x in enumerate(sparse_similarities)
    }

    dense_similarities = [x[0] for x in dense_result_tuples]
    max_dense_sim = max(dense_similarities)
    min_dense_sim = min(dense_similarities)
    dense_similarities = [
        (x - min_dense_sim) / (max_dense_sim - min_dense_sim)
        for x in dense_similarities
    ]
    dense_per_node = {
        dense_result_tuples[i][1].node_id: x
        for i, x in enumerate(dense_similarities)
    }

    fused_similarities = []
    for node_id in all_nodes_dict:
        sparse_sim = sparse_per_node.get(node_id, 0)
        dense_sim = dense_per_node.get(node_id, 0)
        fused_sim = alpha * (sparse_sim + dense_sim)
        fused_similarities.append((fused_sim, all_nodes_dict[node_id]))

    fused_similarities.sort(key=lambda x: x[0], reverse=True)
    fused_similarities = fused_similarities[:top_k]

    return VectorStoreQueryResult(
        nodes=[x[1] for x in fused_similarities],
        similarities=[x[0] for x in fused_similarities],
        ids=[x[1].node_id for x in fused_similarities],
    )

vector_store = QdrantVectorStore(
    "llama2_paper",
    client=client,
    enable_hybrid=True,
    hybrid_fusion_fn=relative_score_fusion,
)

"""
You may have noticed the alpha parameter in the above function. This can be set directely in the `as_query_engine()` call, which will set it in the vector index retriever.
"""
logger.info("You may have noticed the alpha parameter in the above function. This can be set directely in the `as_query_engine()` call, which will set it in the vector index retriever.")

index.as_query_engine(alpha=0.5, similarity_top_k=2)

"""
### Customizing Hybrid Qdrant Collections

Instead of letting llama-index do it, you can also configure your Qdrant hybrid collections ahead of time.

**NOTE:** The names of vector configs must be `text-dense` and `text-sparse` if creating a hybrid index.
"""
logger.info("### Customizing Hybrid Qdrant Collections")


client.recreate_collection(
    collection_name="llama2_paper",
    vectors_config={
        "text-dense": models.VectorParams(
            size=1536,  # openai vector size
            distance=models.Distance.COSINE,
        )
    },
    sparse_vectors_config={
        "text-sparse": models.SparseVectorParams(
            index=models.SparseIndexParams()
        )
    },
)

vector_store = QdrantVectorStore(
    collection_name="llama2_paper", client=client, enable_hybrid=True
)

logger.info("\n\n[DONE]", bright=True)