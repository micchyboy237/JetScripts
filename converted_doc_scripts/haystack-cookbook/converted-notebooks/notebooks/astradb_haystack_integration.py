from haystack import Document, Pipeline
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.generators import OllamaFunctionCallingAdapterGenerator
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.retrievers.astra import AstraEmbeddingRetriever
from haystack_integrations.document_stores.astra import AstraDocumentStore
from jet.logger import CustomLogger
import logging
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
## Introduction

In this notebook, you'll learn how to use [AstraDB](https://docs.datastax.com/en/astra-serverless/docs/) as a data source in your Haystack pipelines.

# Prerequisites

You'll need an [OpenAPI key](https://help.openai.com/en/articles/4936850-where-do-i-find-my-api-key) to follow along. (Haystack is model-agnostic so feel free to use a different one if you'd prefer!)

You'll need the following variables in order to use the Haystack extension. The following tutorials will show you how to create an AstraDB database, and save these pieces of information.

- API Endpoint
- Token
- Astra keyspace
- Astra collection name

Follow the first step in this [this tutorial to create a free AstraDB database](https://docs.datastax.com/en/astra-serverless/docs/manage/db/manage-create.html) and save your database ID, application token, keyspace, and database region.

[Follow these steps to create a collection](https://docs.datastax.com/en/astra/astra-db-vector/databases/manage-collections.html). Save the name of your collection.

Choose the number of dimensions that matches the [embedding model](https://haystack.deepset.ai/blog/what-is-text-vectorization-in-nlp) you plan on using. For this example we'll use a 384-dimension model, [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

Next, install our dependencies.
"""
logger.info("## Introduction")

# !pip install astra-haystack sentence-transformers

"""
Here you'll enter your credentials and such. In production code, you'd want to use environment variables for sensitive credentials such as the application token to avoid committing those to source control.
"""
logger.info("Here you'll enter your credentials and such. In production code, you'd want to use environment variables for sensitive credentials such as the application token to avoid committing those to source control.")

# from getpass import getpass

# os.environ["OPENAI_API_KEY"] = getpass("Enter your openAI key:")
# os.environ["ASTRA_DB_API_ENDPOINT"] = getpass("Enter your Astra API Endpoint:")
# os.environ["ASTRA_DB_APPLICATION_TOKEN"] = getpass("Enter your Astra application token (e.g.AstraCS:xxx ):")
# ASTRA_DB_COLLECTION_NAME = getpass("enter your Astra collection name:")

"""
Next we'll create a Haystack pipeline to create the embeddings and add them into the `AstraDocumentStore`.
"""
logger.info("Next we'll create a Haystack pipeline to create the embeddings and add them into the `AstraDocumentStore`.")





logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

document_store = AstraDocumentStore(
    astra_collection=ASTRA_DB_COLLECTION_NAME,
    duplicates_policy=DuplicatePolicy.SKIP,
    embedding_dim=384,
)


documents = [
    Document(content="There are over 7,000 languages spoken around the world today."),
    Document(
        content="Elephants have been observed to behave in a way that indicates"
        " a high level of self-awareness, such as recognizing themselves in mirrors."
    ),
    Document(
        content="In certain parts of the world, like the Maldives, Puerto Rico, "
        "and San Diego, you can witness the phenomenon of bioluminescent waves."
    ),
]
index_pipeline = Pipeline()
index_pipeline.add_component(
    instance=SentenceTransformersDocumentEmbedder(model=embedding_model_name),
    name="embedder",
)
index_pipeline.add_component(instance=DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP), name="writer")
index_pipeline.connect("embedder.documents", "writer.documents")

index_pipeline.run({"embedder": {"documents": documents}})

logger.debug(document_store.count_documents())

"""
Next we'll make a RAG pipeline so we can query our documents.
"""
logger.info("Next we'll make a RAG pipeline so we can query our documents.")


prompt_template = """
                Given these documents, answer the question.
                Documents:
                {% for doc in documents %}
                    {{ doc.content }}
                {% endfor %}
                Question: {{question}}
                Answer:
                """

rag_pipeline = Pipeline()
rag_pipeline.add_component(
    instance=SentenceTransformersTextEmbedder(model=embedding_model_name),
    name="embedder",
)
rag_pipeline.add_component(instance=AstraEmbeddingRetriever(document_store=document_store), name="retriever")
rag_pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
rag_pipeline.add_component(instance=OllamaFunctionCallingAdapterGenerator(), name="llm")
rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")
rag_pipeline.connect("embedder", "retriever")
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")
rag_pipeline.connect("llm.replies", "answer_builder.replies")
rag_pipeline.connect("llm.meta", "answer_builder.meta")
rag_pipeline.connect("retriever", "answer_builder.documents")


rag_pipeline.draw("./rag_pipeline.png")


question = "How many languages are there in the world today?"
result = rag_pipeline.run(
    {
        "embedder": {"text": question},
        "retriever": {"top_k": 2},
        "prompt_builder": {"question": question},
        "answer_builder": {"query": question},
    }
)

logger.debug(result)

"""
The output should be something like this:
```bash
{'answer_builder': {'answers': [GeneratedAnswer(data='There are over 7,000 languages spoken around the world today.', query='How many languages are there in the world today?', documents=[Document(id=cfe93bc1c274908801e6670440bf2bbba54fad792770d57421f85ffa2a4fcc94, content: 'There are over 7,000 languages spoken around the world today.', score: 0.9267925, embedding: vector of size 384), Document(id=6f20658aeac3c102495b198401c1c0c2bd71d77b915820304d4fbc324b2f3cdb, content: 'Elephants have been observed to behave in a way that indicates a high level of self-awareness, such ...', score: 0.5357444, embedding: vector of size 384)], meta={'model': 'llama3.2-2024-07-18', 'index': 0, 'finish_reason': 'stop', 'usage': {'completion_tokens': 14, 'prompt_tokens': 83, 'total_tokens': 97}})]}}
```

Now that you understand how to use AstraDB as a data source for your Haystack pipeline. Thanks for reading! To learn more about Haystack, [join us on Discord](https://discord.gg/QMP5jgMH) or [sign up for our Monthly newsletter](https://landing.deepset.ai/haystack-community-updates?utm_campaign=developer-relations&utm_source=astradb-haystack-notebook).
"""
logger.info("The output should be something like this:")

logger.info("\n\n[DONE]", bright=True)