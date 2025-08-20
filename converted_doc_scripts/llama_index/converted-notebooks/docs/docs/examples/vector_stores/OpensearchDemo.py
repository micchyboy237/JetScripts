from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Document
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.core.settings import Settings
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.readers.elasticsearch import ElasticsearchReader
from llama_index.vector_stores.opensearch import (
OpensearchVectorStore,
OpensearchVectorClient,
)
from os import getenv
import os
import regex as re
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/OpensearchDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Opensearch Vector Store

Elasticsearch only supports Lucene indices, so only Opensearch is supported.

**Note on setup**: We setup a local Opensearch instance through the following doc. https://opensearch.org/docs/1.0/

If you run into SSL issues, try the following `docker run` command instead: 
```
docker run -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" -e "plugins.security.disabled=true" opensearchproject/opensearch:1.0.1
```

Reference: https://github.com/opensearch-project/OpenSearch/issues/1598

Download Data
"""
logger.info("# Opensearch Vector Store")

# %pip install llama-index-readers-elasticsearch
# %pip install llama-index-vector-stores-opensearch
# %pip install llama-index-embeddings-ollama

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'


endpoint = getenv("OPENSEARCH_ENDPOINT", "http://localhost:9200")
idx = getenv("OPENSEARCH_INDEX", "gpt-index-demo")
documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

text_field = "content"
embedding_field = "embedding"
client = OpensearchVectorClient(
    endpoint, idx, 1536, embedding_field=embedding_field, text_field=text_field
)
vector_store = OpensearchVectorStore(client)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents=documents, storage_context=storage_context
)

query_engine = index.as_query_engine()
res = query_engine.query("What did the author do growing up?")
res.response

"""
The OpenSearch vector store supports [filter-context queries](https://opensearch.org/docs/latest/query-dsl/query-filter-context/).
"""
logger.info("The OpenSearch vector store supports [filter-context queries](https://opensearch.org/docs/latest/query-dsl/query-filter-context/).")


text_chunks = documents[0].text.split("\n\n")

footnotes = [
    Document(
        text=chunk,
        id=documents[0].doc_id,
        metadata={"is_footnote": bool(re.search(r"^\s*\[\d+\]\s*", chunk))},
    )
    for chunk in text_chunks
    if bool(re.search(r"^\s*\[\d+\]\s*", chunk))
]

for f in footnotes:
    index.insert(f)

footnote_query_engine = index.as_query_engine(
    filters=MetadataFilters(
        filters=[
            ExactMatchFilter(
                key="term", value='{"metadata.is_footnote": "true"}'
            ),
            ExactMatchFilter(
                key="query_string",
                value='{"query": "content: space AND content: lisp"}',
            ),
        ]
    )
)

res = footnote_query_engine.query(
    "What did the author about space aliens and lisp?"
)
res.response

"""
## Use reader to check out what VectorStoreIndex just created in our index.

Reader works with Elasticsearch too as it just uses the basic search features.
"""
logger.info("## Use reader to check out what VectorStoreIndex just created in our index.")


rdr = ElasticsearchReader(endpoint, idx)
docs = rdr.load_data(text_field, embedding_field=embedding_field)
logger.debug("embedding dimension:", len(docs[0].embedding))
logger.debug("all fields in index:", docs[0].metadata.keys())

logger.debug("total number of chunks created:", len(docs))

docs = rdr.load_data(text_field, {"query": {"match": {text_field: "Lisp"}}})
logger.debug("chunks that mention Lisp:", len(docs))
docs = rdr.load_data(text_field, {"query": {"match": {text_field: "Yahoo"}}})
logger.debug("chunks that mention Yahoo:", len(docs))

"""
## Hybrid query for opensearch vector store
Hybrid query has been supported since OpenSearch 2.10. It is a combination of vector search and text search. It is useful when you want to search for a specific text and also want to filter the results by vector similarity. You can find more details: https://opensearch.org/docs/latest/query-dsl/compound/hybrid/.

### Prepare Search Pipeline

Create a new [search pipeline](https://opensearch.org/docs/latest/search-plugins/search-pipelines/creating-search-pipeline/) with [score normalization and weighted harmonic mean combination](https://opensearch.org/docs/latest/search-plugins/search-pipelines/normalization-processor/).

```
PUT /_search/pipeline/hybrid-search-pipeline
{
  "description": "Post processor for hybrid search",
  "phase_results_processors": [
    {
      "normalization-processor": {
        "normalization": {
          "technique": "min_max"
        },
        "combination": {
          "technique": "harmonic_mean",
          "parameters": {
            "weights": [
              0.3,
              0.7
            ]
          }
        }
      }
    }
  ]
}
```

### Initialize a OpenSearch client and vector store supporting hybrid query with search pipeline details
"""
logger.info("## Hybrid query for opensearch vector store")


endpoint = getenv("OPENSEARCH_ENDPOINT", "http://localhost:9200")
idx = getenv("OPENSEARCH_INDEX", "auto_retriever_movies")

text_field = "content"
embedding_field = "embedding"
client = OpensearchVectorClient(
    endpoint,
    idx,
    4096,
    embedding_field=embedding_field,
    text_field=text_field,
    search_pipeline="hybrid-search-pipeline",
)


embed_model = OllamaEmbedding(model_name="mxbai-embed-large")

vector_store = OpensearchVectorStore(client)

"""
### Prepare the index
"""
logger.info("### Prepare the index")



storage_context = StorageContext.from_defaults(vector_store=vector_store)

nodes = [
    TextNode(
        text="The Shawshank Redemption",
        metadata={
            "author": "Stephen King",
            "theme": "Friendship",
        },
    ),
    TextNode(
        text="The Godfather",
        metadata={
            "director": "Francis Ford Coppola",
            "theme": "Mafia",
        },
    ),
    TextNode(
        text="Inception",
        metadata={
            "director": "Christopher Nolan",
        },
    ),
]

index = VectorStoreIndex(
    nodes, storage_context=storage_context, embed_model=embed_model
)

"""
### Search the index with hybrid query by specifying the vector store query mode: VectorStoreQueryMode.HYBRID with filters
"""
logger.info("### Search the index with hybrid query by specifying the vector store query mode: VectorStoreQueryMode.HYBRID with filters")


filters = MetadataFilters(
    filters=[
        ExactMatchFilter(
            key="term", value='{"metadata.theme.keyword": "Mafia"}'
        )
    ]
)

retriever = index.as_retriever(
    filters=filters, vector_store_query_mode=VectorStoreQueryMode.HYBRID
)

result = retriever.retrieve("What is inception about?")

logger.debug(result)

logger.info("\n\n[DONE]", bright=True)