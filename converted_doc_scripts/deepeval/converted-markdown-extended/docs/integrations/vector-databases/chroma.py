from deepeval import evaluate
from deepeval.metrics import (
ContextualPrecisionMetric,
ContextualRecallMetric,
ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from jet.logger import logger
from sentence_transformers import SentenceTransformer
import chromadb
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
id: chroma
title: Chroma
sidebar_label: Chroma
---

## Quick Summary

**Chroma** is one of the most popular open-source AI application databases, and supports many retrieval features such as embeddings storage, vector search, document storage, metadata filtering, and multi-modal retrieval.

DeepEval allows you to easily evaluate and optimize your Chroma retriever by **tuning hyperparameters** like `n_results` (more commonly known as top-K) and the `embedding model` used in your Chroma retrieval pipeline.

:::caution
Chroma is not only an optional retriever you can evaluate, it is also a **required dependency** for the `deepeval.synthesizer.generate_goldens_from_docs()` method.
This method uses Chroma as its built-in backend for chunk storage and retrieval during context construction. If you plan to generate goldens from documents, make sure to install `chromadb`:
:::

:::info
To get started, install Chroma through the CLI using the following command:

# pip install chromadb

:::

To learn more about using Chroma for your RAG pipeline, [visit this page](https://www.trychroma.com/). The diagram below illustrates how you can utilize Chroma as the entire retrieval pipeline for your LLM application.

<div
  style={{
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "column",
  }}
>
  <img
    src="https://www.trychroma.com/_next/static/media/computer.fcd1bd54.svg"
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
    Source: Chroma
  </div>
</div>

## Setup Chroma

To get started with **Chroma**, initialize a persistent client and create a collection to store your documents. The collection acts as a vector database for storing and retrieving embeddings, while the persistent client ensures data is retained across sessions.
"""
logger.info("## Quick Summary")


client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="rag_documents")

"""
Next, define an **embedding model** (we'll use `sentence_transformers`) to convert document chunks into vectors before adding them to your Chroma collection, along with the document chunks as metadata.
"""
logger.info("Next, define an **embedding model** (we'll use `sentence_transformers`) to convert document chunks into vectors before adding them to your Chroma collection, along with the document chunks as metadata.")

...

model = SentenceTransformer("all-MiniLM-L6-v2")

document_chunks = [
    "Chroma is an open-source vector database for efficient embedding retrieval.",
    "It enables fast semantic search using vector similarity.",
    "Chroma retrieves relevant data with cosine similarity.",
    ...
]

for i, chunk in enumerate(document_chunks):
    embedding = model.encode(chunk).tolist()  # Convert text to vector
    collection.add(
        ids=[str(i)],  # Unique ID for each document
        embeddings=[embedding],  # Vector representation
        metadatas=[{"text": chunk}]  # Store original text as metadata
    )

"""
You'll be querying from this Chroma collection during generation to retrieve relevant contexts based on the user `input`, before passing them along with your input into your LLM's prompt template.

:::note
By default, Chroma utilizes `cosine similarity` to find similar chunks.
:::

## Evaluating Chroma Retrieval

To evaluate your Chroma retriever, you'll first need to prepare an `input` query and generate a response from your RAG pipeline in order to create an `LLMTestCase`. You'll also need to extract the contexts retrieved from your Chroma collection during generation and prepare the expected LLM response to complete the `LLMTestCase`.

:::information
By default, `input` and `actual_output` are required for all metrics. However, `retrieval_context`, `context`, and `expected_output` are optional, and different metrics may or may not require additional parameters. To check the specific requirements, [visit the metrics section](/docs/metrics-introduction).
:::

After you've prepared your `LLMTestCase`, evaluating your Chroma retriever is as easy passing the test case along with your selection of metrics into DeepEval's `evaluate` function.

### Preparing your Test Case

To prepare our test case, we'll be using `"How does Chroma work?"` as our input. Before generating a response from your RAG pipeline, you'll first need to retrieve the relevant context using a `search` function. Our `search` function in the example below first embeds the input query before retrieving the top three most relevant text chunks (`n_results=3`) from our chroma collection.
"""
logger.info("## Evaluating Chroma Retrieval")

...

def search(query):
    query_embedding = model.encode(query).tolist()

    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=3  # Retrieve top-K matches
    )

    return res["metadatas"][0][0]["text"] if res["metadatas"][0] else None

query = "How does Chroma work?"
retrieval_context = search(query)

"""
Next, we'll pass the retrieved context from our Chroma collection into the LLM's prompt template to generate the final response.
"""
logger.info("Next, we'll pass the retrieved context from our Chroma collection into the LLM's prompt template to generate the final response.")

...

prompt = """
Answer the user question based on the supporting context.

User Question:
{input}

Supporting Context:
{retrieval_context}
"""

actual_output = generate(prompt)  # Replace with your LLM function
logger.debug(actual_output)
logger.debug(expected_output)

"""
Printing the `actual_output` generated by our RAG pipeline yields the following example:

Chroma is a lightweight vector database designed for AI applications, enabling fast semantic retrieval.

Let's compare this to the `expected_output` we've prepared:

Chroma is an open-source vector database that enables fast retrieval using cosine similarity.

With all the elements ready, we'll create an `LLMTestCase` by providing the input and expected output, along with the actual output and retrieved context.
"""
logger.info("Printing the `actual_output` generated by our RAG pipeline yields the following example:")


...

test_case = LLMTestCase(
    input=input,
    actual_output=actual_output,
    retrieval_context=retrieval_context,
    expected_output=expected_output
)

"""
### Running Evaluations

To begin running evaluations, we'll need to define metrics relevant to our Chroma retriever. These include `ContextualRecallMetric`, `ContextualPrecisionMetric`, and `ContextualRelevancyMetric`, which specifically evaluate RAG retrievers.

:::tip
To learn more about how these metrics are calculated and why they're relevant to retrievers, visit the [individual metric pages](/docs/metrics-contextual-precision).
:::
"""
logger.info("### Running Evaluations")


contextual_precision = ContextualPrecisionMetric()
contextual_recall = ContextualRecallMetric(),
contextual_relevancy = ContextualRelevancyMetric()

"""
To run evaluations, simply pass the prepared test case you've prepared into the `evaluate` function, along with the retriever metrics you defined.


...

evaluate(
    [test_case],
    metrics=[contextual_recall, contextual_precision, contextual_relevancy]
)

## Improving Chroma Retrieval

Hypothetically, we've run multiple inputs and prepared several test cases, consistently observing that the `Contextual Relevancy` score is below the required threshold.

| <div style={{width: "350px"}}>Inputs</div> | <div style={{width: "250px"}}>Contextual Relevancy Score</div> | <div style={{width: "250px"}}>Contextual Recall Score</div> |
| ------------------------------------------ | -------------------------------------------------------------- | ----------------------------------------------------------- |
| "How does Chroma work?"                    | 0.45                                                           | 0.85                                                        |
| "What is the retrieval process in Chroma?" | 0.43                                                           | 0.92                                                        |
| "Explain Chroma's vector database."        | 0.55                                                           | 0.67                                                        |

This suggests that you may need to adjust the length of each document or tweak `n_results` to retrieve more relevant contexts from your Chroma collection. This is because Contextual Relevancy evaluates both the **retrieved text chunks and the top-K selection**.

:::tip
If you're curious about which metrics evaluate which specific retrieval parameters, [check out this guide](/guides/guides-rag-evaluation).
:::

Depending on the failing scores in your retriever, you'll want to experiment with different parameters (e.g., `n_results`, `embedding model`, etc.) in your Chroma retrieval pipeline until you're satisfied with the results. This can be as simple as writing a for loop to run evaluations many times:
"""
logger.info("## Improving Chroma Retrieval")

...

def search(query, n_results):
    query_embedding = model.encode(query).tolist()

    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results  # Retrieve top-K matches
    )

    return res["metadatas"][0][0]["text"] if res["metadatas"][0] else None


...

for top_k in [3, 5, 7]:
    retrieval_context = search(input_query, top_k)

    ...

    evaluate(
        [test_case],
        metrics=[contextual_recall, contextual_precision, contextual_relevancy]
    )

"""
:::note
If you need a systematic way to analyze your retriever and compare the effects of changing chroma hyperparameters side by side, you'll want to [log in to Confident AI](https://www.confident-ai.com/).
:::
"""
logger.info("If you need a systematic way to analyze your retriever and compare the effects of changing chroma hyperparameters side by side, you'll want to [log in to Confident AI](https://www.confident-ai.com/).")

logger.info("\n\n[DONE]", bright=True)