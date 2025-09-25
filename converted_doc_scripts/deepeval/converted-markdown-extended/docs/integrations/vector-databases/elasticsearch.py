from deepeval import evaluate
from deepeval.metrics import (
ContextualRecallMetric,
ContextualPrecisionMetric,
ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from elasticsearch import Elasticsearch
from jet.logger import logger
from sentence_transformers import SentenceTransformer
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
---
id: elasticsearch
title: Elasticsearch
sidebar_label: Elasticsearch
---

## Quick Summary

DeepEval allows you to evaluate your **Elasticsearch** retriever and optimize retrieval hyperparameters like `top-K`, `embedding model`, and `similarity function`.

:::info
To get started, install Elasticsearch through the CLI using the following command:

# pip install elasticsearch

:::
Elasticsearch is a fast and scalable search engine that works as a high-performance vector database for RAG applications. It handles **large-scale retrieval workloads** efficiently, making it ideal for production use. Learn more about Elasticsearch [here](https://www.elastic.co/guide/en/elasticsearch/reference/current/getting-started.html).

This diagram illustrates how the Elasticsearch retriever fits into your RAG pipeline.

<div
  style={{
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "column",
  }}
>
  <img
    src="https://images.contentstack.io/v3/assets/bltefdd0b53724fa2ce/blt1496b19e4c6f9e66/66ba412a46b3f4241b969f48/rag-in-action.jpeg"
    style={{
      margin: "10px",
      height: "auto",
      maxHeight: "800px",
    }}
  />
  <div
    style={{
      fontSize: "13px",
    }}
  >
    Source: Elasticsearch
  </div>
</div>

## Setup Elasticsearch

To get started, connect to your local Elastic cluster using the `"elastic"` username and the `ELASTIC_PASSWORD` environment variable.
"""
logger.info("## Quick Summary")


username = 'elastic'
password = os.getenv('ELASTIC_PASSWORD') # Value you set in the environment variable

client = Elasticsearch(
    "http://localhost:9200",
    basic_auth=(username, password)
)

"""
Next, create an Elasticsearch index with the appropriate type mappings to store `text` and `embedding` as a `dense_vector`.
"""
logger.info("Next, create an Elasticsearch index with the appropriate type mappings to store `text` and `embedding` as a `dense_vector`.")

if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body={
        "mappings": {
            "properties": {
                "text": {"type": "text"},  # Stores chunk text
                "embedding": {"type": "dense_vector", "dims": 384}  # Stores embeddings
            }
        }
    })

"""
Finally, define an embedding model to convert your document chunks into vectors before indexing them in Elasticsearch for retrieval.
"""
logger.info("Finally, define an embedding model to convert your document chunks into vectors before indexing them in Elasticsearch for retrieval.")

model = SentenceTransformer("all-MiniLM-L6-v2")

document_chunks = [
    "Elasticsearch is a distributed search engine.",
    "RAG improves AI-generated responses with retrieved context.",
    "Vector search enables high-precision semantic retrieval.",
    ...
]

for i, chunk in enumerate(document_chunks):
    embedding = model.encode(chunk).tolist()  # Convert text to vector
    es.index(index=index_name, id=i, body={"text": chunk, "embedding": embedding})

"""
To use Elasticsearch as part of your RAG pipeline, simply use it to retrieve relevant contexts and insert them into your prompt template for generation. This ensures your model has the necessary context to generate accurate and informed responses.

## Evaluating Elasticsearch Retrieval

Evaluating your Elasticsearch retriever consists of 2 steps:

1. Preparing an `input` query along with the expected LLM response, and using the `input` to generate a response from your RAG pipeline to create an `LLMTestCase` containing the input, actual output, expected output, and retrieval context.
2. Evaluating the test case using a selection of retrieval metrics.

:::information
An `LLMTestCase` allows you to create unit tests for your LLM applications, helping you identify specific weaknesses in your RAG application.
:::

### Preparing your Test Case

Since the first step in generating a response from your RAG pipeline is retrieving the relevant `retrieval_context` from your Elasticsearch index, first perform this retrieval for your `input` query.
"""
logger.info("## Evaluating Elasticsearch Retrieval")

def search(query):
    query_embedding = model.encode(query).tolist()

    res = es.search(index=index_name, body={
        "knn": {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": 3  # Retrieve the top match
            "num_candidates": 10  # Controls search speed vs accuracy
        }
    })

    return res["hits"]["hits"][0]["_source"]["text"] if res["hits"]["hits"] else None

query = "How does Elasticsearch work?"
retrieval_context = search(query)

"""
Next, pass the retrieved context into your LLM's prompt template to generate a response.
"""
logger.info("Next, pass the retrieved context into your LLM's prompt template to generate a response.")

prompt = """
Answer the user question based on the supporting context

User Question:
{input}

Supporting Context:
{retrieval_context}
"""

actual_output = generate(prompt) # hypothetical function, replace with your own LLM
logger.debug(actual_output)

"""
Let's examine the `actual_output` generated by our RAG pipeline:

Elasticsearch indexes document chunks using an inverted index for fast full-text search and retrieval.

Finally, create an `LLMTestCase` using the input and expected output you prepared, along with the actual output and retrieval context you generated.
"""
logger.info("Let's examine the `actual_output` generated by our RAG pipeline:")


test_case = LLMTestCase(
    input=input,
    actual_output=actual_output,
    retrieval_context=retrieval_context,
    expected_output="Elasticsearch uses inverted indexes for keyword searches and dense vector similarity for semantic search.",
)

"""
### Running Evaluations

To run evaluations on the `LLMTestCase`, we first need to define relevant `deepeval` metrics to evaluate the Elasticsearch retriever: contextual recall, contextual precision, and contextual relevancy.

:::note
These **contextual metrics** help assess your retriever. For more retriever evaluation details, check out this [guide](/guides/guides-rag-evaluation).  
:::
"""
logger.info("### Running Evaluations")


contextual_recall = ContextualRecallMetric(),
contextual_precision = ContextualPrecisionMetric()
contextual_relevancy = ContextualRelevancyMetric()

"""
Finally, pass the test case and metrics into the `evaluate` function to begin the evaluation.


evaluate(
    [test_case],
    metrics=[contextual_recall, contextual_precision, contextual_relevancy]
)

## Improving Elasticsearch Retrieval

Below is a table outlining the hypothetical metric scores for your evaluation run.

| <div style={{width: "350px"}}>Metric</div> | <div style={{width: "350px"}}>Score</div> |
| ------------------------------------------ | ----------------------------------------- |
| Contextual Precision                       | 0.85                                      |
| Contextual Recall                          | 0.92                                      |
| Contextual Relevancy                       | 0.44                                      |

:::info
Each contextual metric evaluates a **specific hyperparameter**. To learn more about this, read [this guide on RAG evaluation](/guides/guides-rag-evaluation).
:::

To improve your Elasticsearch retriever, you'll need to experiment with various hyperparameters and prepare `LLMTestCase`s using generations from different retriever versions.

Ultimately, analyzing improvements and regressions in **contextual metric scores** (the three metrics defined above) will help you determine the optimal hyperparameter combination for your Elasticsearch retriever.

:::tip
For a more detailed guide on tuning your retriever’s hyperparameters, check out [this guide](/guides/guides-optimizing-hyperparameters).
:::
"""
logger.info("## Improving Elasticsearch Retrieval")

logger.info("\n\n[DONE]", bright=True)