from haystack import Document, Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import OllamaDocumentEmbedder, OllamaTextEmbedder
from haystack.components.generators import OllamaGenerator
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.retrievers.mongodb_atlas import (
MongoDBAtlasEmbeddingRetriever,
)
from haystack_integrations.document_stores.mongodb_atlas import (
MongoDBAtlasDocumentStore,
)
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/rag/Haystack_MongoDB_Atlas_RAG.ipynb)

# Haystack and MongoDB Atlas RAG notebook

Install dependencies:
"""
logger.info("# Haystack and MongoDB Atlas RAG notebook")

pip install haystack-ai mongodb-atlas-haystack tiktoken

"""
## Setup MongoDB Atlas connection and Open AI


* Set the MongoDB connection string. Follow the steps [here](https://www.mongodb.com/docs/manual/reference/connection-string/) to get the connection string from the Atlas UI.

* Set the Ollama API key. Steps to obtain an API key as [here](https://help.ollama.com/en/articles/4936850-where-do-i-find-my-ollama-api-key)
"""
logger.info("## Setup MongoDB Atlas connection and Open AI")

# import getpass

# os.environ["MONGO_CONNECTION_STRING"] = getpass.getpass(
    "Enter your MongoDB connection string:"
)

# os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your Open AI Key:")

"""


## Create vector search index on collection

Follow this [tutorial](https://www.mongodb.com/docs/atlas/atlas-vector-search/create-index/) to create a vector index on database: `haystack_test` collection `test_collection`.

Verify that the index name is `vector_index` and the syntax specify:
```
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    }
  ]
}
```

### Setup vector store to load documents:
"""
logger.info("## Create vector search index on collection")


documents = [
    Document(content="My name is Jean and I live in Paris."),
    Document(content="My name is Mark and I live in Berlin."),
    Document(content="My name is Giorgio and I live in Rome."),
]

document_store = MongoDBAtlasDocumentStore(
    database_name="haystack_test",
    collection_name="test_collection",
    vector_search_index="vector_index",
)

"""
Build the writer pipeline to load documnets
"""
logger.info("Build the writer pipeline to load documnets")

doc_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)

doc_embedder = OllamaDocumentEmbedder()

indexing_pipe = Pipeline()
indexing_pipe.add_component(instance=doc_embedder, name="doc_embedder")
indexing_pipe.add_component(instance=doc_writer, name="doc_writer")

indexing_pipe.connect("doc_embedder.documents", "doc_writer.documents")

indexing_pipe.run({"doc_embedder": {"documents": documents}})

"""
## Build a RAG Pipeline

Lets create a pipeline that will Retrieve Augment and Generate a response for user questions
"""
logger.info("## Build a RAG Pipeline")

prompt_template = """
    You are an assistant allowed to use the following context documents.\nDocuments:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    \Query: {{query}}
    \nAnswer:
"""

rag_pipeline = Pipeline()
rag_pipeline.add_component("text_embedder", OllamaTextEmbedder())

rag_pipeline.add_component(
    instance=MongoDBAtlasEmbeddingRetriever(document_store=document_store, top_k=15),
    name="retriever",
)

rag_pipeline.add_component(
    instance=PromptBuilder(template=prompt_template), name="prompt_builder"
)

rag_pipeline.add_component(instance=OllamaGenerator(), name="llm")

rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")

"""
Lets test the pipeline
"""
logger.info("Lets test the pipeline")

query = "Where does mark live?"
result = rag_pipeline.run(
    {
        "text_embedder": {"text": query},
        "prompt_builder": {"query": query},
    }
)
logger.debug(result["llm"]["replies"][0])

logger.info("\n\n[DONE]", bright=True)