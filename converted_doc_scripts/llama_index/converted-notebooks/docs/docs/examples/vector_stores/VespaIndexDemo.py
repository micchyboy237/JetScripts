from IPython.display import Markdown, display
from jet.logger import CustomLogger
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import (
FilterOperator,
FilterCondition,
MetadataFilter,
MetadataFilters,
)
from llama_index.core.vector_stores.types import (
VectorStoreQuery,
VectorStoreQueryMode,
)
from llama_index.vector_stores.vespa import VespaVectorStore
from vespa.package import (
ApplicationPackage,
Field,
Schema,
Document,
HNSW,
RankProfile,
Component,
Parameter,
FieldSet,
GlobalPhaseRanking,
Function,
)
import openai
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://vespa.ai/assets/vespa-ai-logo-heather.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://vespa.ai/assets/vespa-ai-logo-rock.svg">
  <img alt="#Vespa" width="200" src="https://vespa.ai/assets/vespa-ai-logo-rock.svg" style="margin-bottom: 25px;">
</picture>

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/VespaIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Vespa Vector Store demo

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Vespa Vector Store demo")

# %pip install llama-index-vector-stores-vespa llama-index pyvespa

"""
#### Setting up API key
"""
logger.info("#### Setting up API key")


# os.environ["OPENAI_API_KEY"] = "sk-..."
# openai.api_key = os.environ["OPENAI_API_KEY"]

"""
#### Load documents, build the VectorStoreIndex
"""
logger.info("#### Load documents, build the VectorStoreIndex")


"""
## Defining some sample data

Let's insert some documents.
"""
logger.info("## Defining some sample data")


nodes = [
    TextNode(
        text="The Shawshank Redemption",
        metadata={
            "author": "Stephen King",
            "theme": "Friendship",
            "year": 1994,
        },
    ),
    TextNode(
        text="The Godfather",
        metadata={
            "director": "Francis Ford Coppola",
            "theme": "Mafia",
            "year": 1972,
        },
    ),
    TextNode(
        text="Inception",
        metadata={
            "director": "Christopher Nolan",
            "theme": "Fiction",
            "year": 2010,
        },
    ),
    TextNode(
        text="To Kill a Mockingbird",
        metadata={
            "author": "Harper Lee",
            "theme": "Mafia",
            "year": 1960,
        },
    ),
    TextNode(
        text="1984",
        metadata={
            "author": "George Orwell",
            "theme": "Totalitarianism",
            "year": 1949,
        },
    ),
    TextNode(
        text="The Great Gatsby",
        metadata={
            "author": "F. Scott Fitzgerald",
            "theme": "The American Dream",
            "year": 1925,
        },
    ),
    TextNode(
        text="Harry Potter and the Sorcerer's Stone",
        metadata={
            "author": "J.K. Rowling",
            "theme": "Fiction",
            "year": 1997,
        },
    ),
]

"""
### Initilizing the VespaVectorStore

To make it really simple to get started, we provide a template Vespa application that will be deployed upon initializing the vector store.

This is a huge abstraction and there are endless opportunities to tailor and customize the Vespa application to your needs. But for now, let's keep it simple and initialize with the default template.
"""
logger.info("### Initilizing the VespaVectorStore")


vector_store = VespaVectorStore()
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context)

"""
### Deleting documents
"""
logger.info("### Deleting documents")

node_to_delete = nodes[0].node_id
node_to_delete

vector_store.delete(ref_doc_id=node_to_delete)

"""
## Querying
"""
logger.info("## Querying")


query = VectorStoreQuery(
    query_str="Great Gatsby",
    mode=VectorStoreQueryMode.TEXT_SEARCH,
    similarity_top_k=1,
)
result = vector_store.query(query)

result

"""
## As retriever

### Default query mode (text search)
"""
logger.info("## As retriever")

retriever = index.as_retriever(vector_store_query_mode="default")
results = retriever.retrieve("Who directed inception?")
display(Markdown(f"**Retrieved nodes:**\n {results}"))

retriever = index.as_retriever(vector_store_query_mode="semantic_hybrid")
results = retriever.retrieve("Who wrote Harry Potter?")
display(Markdown(f"**Retrieved nodes:**\n {results}"))

"""
### As query engine
"""
logger.info("### As query engine")

query_engine = index.as_query_engine()
response = query_engine.query("Who directed inception?")
display(Markdown(f"**Response:** {response}"))

query_engine = index.as_query_engine(
    vector_store_query_mode="semantic_hybrid", verbose=True
)
response = query_engine.query(
    "When was the book about the wizard boy published and what was it called?"
)
display(Markdown(f"**Response:** {response}"))
display(Markdown(f"**Sources:** {response.source_nodes}"))

"""
## Using metadata filters

**NOTE**: This metadata filtering is done by llama-index, outside of vespa. For native and much more performant filtering, you should use Vespa's own filtering capabilities.

See [Vespa's documentation](https://docs.vespa.ai/en/reference/query-language-reference.html) for more information.
"""
logger.info("## Using metadata filters")



filters = MetadataFilters(
    filters=[
        MetadataFilter(key="theme", value="Fiction"),
        MetadataFilter(key="year", value=1997, operator=FilterOperator.GT),
    ],
    condition=FilterCondition.OR,
)

retriever = index.as_retriever(filters=filters)
result = retriever.retrieve("Harry Potter")
display(Markdown(f"**Result:** {result}"))

"""
## Abstraction level of this integration

To make it really simple to get started, we provide a template Vespa application that will be deployed upon initializing the vector store. This removes some of the complexity of setting up Vespa for the first time, but for serious use cases, we strongly recommend that you read the [Vespa documentation](docs.vespa.ai) and tailor the application to your needs.

### The template

The provided template Vespa application can be seen below:

```python

hybrid_template = ApplicationPackage(
    name="hybridsearch",
    schema=[
        Schema(
            name="doc",
            document=Document(
                fields=[
                    Field(name="id", type="string", indexing=["summary"]),
                    Field(name="metadata", type="string", indexing=["summary"]),
                    Field(
                        name="text",
                        type="string",
                        indexing=["index", "summary"],
                        index="enable-bm25",
                        bolding=True,
                    ),
                    Field(
                        name="embedding",
                        type="tensor<float>(x[384])",
                        indexing=[
                            "input text",
                            "embed",
                            "index",
                            "attribute",
                        ],
                        ann=HNSW(distance_metric="angular"),
                        is_document_field=False,
                    ),
                ]
            ),
            fieldsets=[FieldSet(name="default", fields=["text", "metadata"])],
            rank_profiles=[
                RankProfile(
                    name="bm25",
                    inputs=[("query(q)", "tensor<float>(x[384])")],
                    functions=[Function(name="bm25sum", expression="bm25(text)")],
                    first_phase="bm25sum",
                ),
                RankProfile(
                    name="semantic",
                    inputs=[("query(q)", "tensor<float>(x[384])")],
                    first_phase="closeness(field, embedding)",
                ),
                RankProfile(
                    name="fusion",
                    inherits="bm25",
                    inputs=[("query(q)", "tensor<float>(x[384])")],
                    first_phase="closeness(field, embedding)",
                    global_phase=GlobalPhaseRanking(
                        expression="reciprocal_rank_fusion(bm25sum, closeness(field, embedding))",
                        rerank_count=1000,
                    ),
                ),
            ],
        )
    ],
    components=[
        Component(
            id="e5",
            type="hugging-face-embedder",
            parameters=[
                Parameter(
                    "transformer-model",
                    {
                        "url": "https://github.com/vespa-engine/sample-apps/raw/master/simple-semantic-search/model/e5-small-v2-int8.onnx"
                    },
                ),
                Parameter(
                    "tokenizer-model",
                    {
                        "url": "https://raw.githubusercontent.com/vespa-engine/sample-apps/master/simple-semantic-search/model/tokenizer.json"
                    },
                ),
            ],
        )
    ],
)
```

Note that the fields `id`, `metadata`, `text`, and `embedding` are required for the integration to work.
The schema name must also be `doc`, and the rank profiles must be named `bm25`, `semantic`, and `fusion`.

Other than that you are free to modify as you see fit by switching out embedding models, adding more fields, or changing the ranking expressions.

For more details, check out this Pyvespa example notebook on [hybrid search](https://pyvespa.readthedocs.io/en/latest/getting-started-pyvespa.html).
"""
logger.info("## Abstraction level of this integration")

logger.info("\n\n[DONE]", bright=True)