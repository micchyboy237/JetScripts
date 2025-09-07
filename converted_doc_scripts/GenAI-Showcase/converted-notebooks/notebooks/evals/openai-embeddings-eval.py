from datasets import load_dataset
from jet.logger import CustomLogger
from ollama import Ollama
from sentence_transformers.util import cos_sim
from tqdm.auto import tqdm
from typing import List
import numpy as np
import os
import pandas as pd
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/evals/ollama-embeddings-eval.ipynb)

[![View Article](https://img.shields.io/badge/View%20Article-blue)](https://www.mongodb.com/developer/products/atlas/choose-embedding-model-rag/?utm_campaign=devrel&utm_source=cross-post&utm_medium=organic_social&utm_content=https%3A%2F%2Fgithub.com%2Fmongodb-developer%2FGenAI-Showcase&utm_term=apoorva.joshi)

# RAG Series Part 1: How to choose the right embedding model for your RAG application

This notebook evaluates the [text-embedding-3-large](https://ollama.com/blog/new-embedding-models-and-api-updates) model.

## Step 1: Install required libraries

- **datasets**: Python library to get access to datasets available on Hugging Face Hub
<p>
- **ollama**: Python library to interact with Ollama APIs
<p>
- **sentence-transformers**: Framework for working with text and image embeddings
<p>
- **numpy**: Python library that provides tools to perform mathematical operations on arrays
<p>
- **pandas**: Python library for data analysis, exploration and manipulation
<p>
- **tdqm**: Python module to show a progress meter for loops
"""
logger.info("# RAG Series Part 1: How to choose the right embedding model for your RAG application")

# ! pip install -qU datasets ollama sentence-transformers numpy pandas tqdm

"""
## Step 2: Setup pre-requisites

Set Ollama API key as environment variable, and initialize the Ollama client.

Steps to obtain an Ollama API Key can be found [here](https://help.ollama.com/en/articles/4936850-where-do-i-find-my-ollama-api-key)
"""
logger.info("## Step 2: Setup pre-requisites")

# import getpass


# os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:")
ollama_client = Ollama()

"""
## Step 3: Download the evaluation dataset

We will use MongoDB's [cosmopedia-wikihow-chunked](https://huggingface.co/datasets/MongoDB/cosmopedia-wikihow-chunked) dataset, which has chunked versions of WikiHow articles from the [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) dataset released by Hugging Face. The dataset is pretty large, so we will only grab the first 25k records for testing.
"""
logger.info("## Step 3: Download the evaluation dataset")


data = load_dataset("MongoDB/cosmopedia-wikihow-chunked", split="train", streaming=True)
data_head = data.take(25000)
df = pd.DataFrame(data_head)

"""
## Step 4: Data analysis

Make sure the length of the dataset is what we expect (25k), preview the data, drop Nones etc.
"""
logger.info("## Step 4: Data analysis")

len(df)

df.head()

df = df[df["text"].notna()]

df.doc_id.nunique()

"""
## Step 5: Creating embeddings

Define the embedding function, and run a quick test.
"""
logger.info("## Step 5: Creating embeddings")


def get_embeddings(
    docs: List[str], model: str = "text-embedding-3-large"
) -> List[List[float]]:
    """
    Generate embeddings using the Ollama API.

    Args:
        docs (List[str]): List of texts to embed
        model (str, optional): Model name. Defaults to "text-embedding-3-large".

    Returns:
        List[float]: Array of embeddings
    """
    docs = [doc.replace("\n", " ") for doc in docs]
    response = ollama_client.embeddings.create(input=docs, model=model)
    response = [r.embedding for r in response.data]
    return response

test_ollama_embed = get_embeddings([df.iloc[0]["text"]])

len(test_ollama_embed[0])

"""
## Step 6: Evaluation

### Measuring embedding latency

Create a local vector store (list) of embeddings for the entire dataset.
"""
logger.info("## Step 6: Evaluation")


texts = df["text"].tolist()

batch_size = 128

embeddings = []
for i in tqdm(range(0, len(texts), batch_size)):
    end = min(len(texts), i + batch_size)
    batch = texts[i:end]
    batch_embeddings = get_embeddings(batch)
    embeddings.extend(np.array(batch_embeddings))

"""
### Measuring retrieval quality

- Create embedding for the user query
<p>
- Get the top 5 most similar documents from the local vector store using cosine similarity as the similarity metric
"""
logger.info("### Measuring retrieval quality")


embeddings = np.asarray(embeddings)

def query(query: str, top_k: int = 3) -> None:
    """
    Query the local vector store for the top 3 most relevant documents.

    Args:
        query (str): User query
        top_k (int, optional): Number of documents to return. Defaults to 3.
    """
    query_emb = np.asarray(get_embeddings([query]))
    scores = cos_sim(query_emb, embeddings)[0]
    idxs = np.argsort(-scores)[:top_k]

    logger.debug(f"Query: {query}")
    for idx in idxs:
        logger.debug(f"Score: {scores[idx]:.4f}")
        logger.debug(texts[idx])
        logger.debug("--------")

query("Give me some tips to improve my mental health.")

query_emb = query("Give me some tips for writing good code.")

query("How do I create a basic webpage?")

query(
    "What are some environment-friendly practices I can incorporate in everyday life?"
)

logger.info("\n\n[DONE]", bright=True)