from datasets import Dataset
from datasets import load_dataset
from datetime import datetime
from jet.llm.ollama.base_langchain import ChatOllama
from jet.llm.ollama.base_langchain import OllamaEmbeddings
from jet.logger import CustomLogger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.base import RunnableSequence
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_mongodb import MongoDBAtlasVectorSearch
from ollama import Ollama
from pymongo import MongoClient
from ragas import RunConfig, evaluate
from ragas.metrics import answer_correctness, answer_similarity
from ragas.metrics import answer_relevancy, faithfulness
from ragas.metrics import context_precision, context_recall
from tqdm.auto import tqdm
from typing import List
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/evals/ragas-evaluation.ipynb)

[![View Article](https://img.shields.io/badge/View%20Article-blue)](https://www.mongodb.com/developer/products/atlas/evaluate-llm-applications-rag/?utm_campaign=devrel&utm_source=cross-post&utm_medium=organic_social&utm_content=https%3A%2F%2Fgithub.com%2Fmongodb-developer%2FGenAI-Showcase&utm_term=apoorva.joshi)

# RAG Series Part 2: How to evaluate your RAG application

This notebook shows how to evaluate a RAG application using the [RAGAS](https://docs.ragas.io/en/stable/index.html) framework.

## Step 1: Install required libraries

- **datasets**: Python library to get access to datasets available on Hugging Face Hub
<p>
- **ragas**: Python library for the RAGAS framework
<p>
- **langchain**: Python library to develop LLM applications using LangChain
<p>
- **langchain-mongodb**: Python package to use MongoDB Atlas vector Search with LangChain
<p>
- **langchain-ollama**: Python package to use Ollama models in LangChain
<p>
- **pymongo**: Python driver to interacting with MongoDB
<p>
- **pandas**: Python library for data analysis, exploration and manipulation
<p>
- **tdqm**: Python module to show a progress meter for loops
<p>
- **matplotlib, seaborn**: Python libraries for data visualization
"""
logger.info("# RAG Series Part 2: How to evaluate your RAG application")

# ! pip install -qU datasets ragas langchain langchain-mongodb langchain-ollama \
pymongo pandas tqdm matplotlib seaborn

"""
## Step 2: Setup pre-requisites

- Set the MongoDB connection string. Follow the steps [here](https://www.mongodb.com/docs/manual/reference/connection-string/) to get the connection string from the Atlas UI.

- Set the Ollama API key. Steps to obtain an API key as [here](https://help.ollama.com/en/articles/4936850-where-do-i-find-my-ollama-api-key)
"""
logger.info("## Step 2: Setup pre-requisites")

# import getpass


# os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your Ollama API Key:")
ollama_client = Ollama()

# MONGODB_URI = getpass.getpass("Enter your MongoDB connection string:")

"""
## Step 3: Download the evaluation dataset
"""
logger.info("## Step 3: Download the evaluation dataset")


data = load_dataset("explodinggradients/ragas-wikiqa", split="train")
df = pd.DataFrame(data)

df.head(1)

len(df)

"""
## Step 4: Create reference document chunks
"""
logger.info("## Step 4: Create reference document chunks")



text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", keep_separator=False, chunk_size=200, chunk_overlap=30
)

def split_texts(texts: List[str]) -> List[str]:
    """
    Split large texts into chunks

    Args:
        texts (List[str]): List of reference texts

    Returns:
        List[str]: List of chunked texts
    """
    chunked_texts = []
    for text in texts:
        chunks = text_splitter.create_documents([text])
        chunked_texts.extend([chunk.page_content for chunk in chunks])
    return chunked_texts

df["chunks"] = df["context"].apply(lambda x: split_texts(x))

all_chunks = df["chunks"].tolist()
docs = [item for chunk in all_chunks for item in chunk]

len(docs)

docs[100]

"""
## Step 5: Create embeddings and ingest them into MongoDB
"""
logger.info("## Step 5: Create embeddings and ingest them into MongoDB")


def get_embeddings(docs: List[str], model: str) -> List[List[float]]:
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

client = MongoClient(MONGODB_URI, appname="devrel.showcase.ragas_eval")
DB_NAME = "ragas_evals"
db = client[DB_NAME]

batch_size = 128

EVAL_EMBEDDING_MODELS = ["text-embedding-ada-002", "mxbai-embed-large"]

for model in EVAL_EMBEDDING_MODELS:
    embedded_docs = []
    logger.debug(f"Getting embeddings for the {model} model")
    for i in tqdm(range(0, len(docs), batch_size)):
        end = min(len(docs), i + batch_size)
        batch = docs[i:end]
        batch_embeddings = get_embeddings(batch, model)
        batch_embedded_docs = [
            {"text": batch[i], "embedding": batch_embeddings[i]}
            for i in range(len(batch))
        ]
        embedded_docs.extend(batch_embedded_docs)
    logger.debug(f"Finished getting embeddings for the {model} model")

    logger.debug(f"Inserting embeddings for the {model} model")
    collection = db[model]
    collection.delete_many({})
    collection.insert_many(embedded_docs)
    logger.debug(f"Finished inserting embeddings for the {model} model")

"""
## Step 6: Compare embedding models for retrieval
"""
logger.info("## Step 6: Compare embedding models for retrieval")

# import nest_asyncio

# nest_asyncio.apply()

def get_retriever(model: str, k: int) -> VectorStoreRetriever:
    """
    Given an embedding model and top k, get a vector store retriever object

    Args:
        model (str): Embedding model to use
        k (int): Number of results to retrieve

    Returns:
        VectorStoreRetriever: A vector store retriever object
    """
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    vector_store = MongoDBAtlasVectorSearch.from_connection_string(
        connection_string=MONGODB_URI,
        namespace=f"{DB_NAME}.{model}",
        embedding=embeddings,
        index_name="vector_index",
        text_key="text",
    )

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )
    return retriever

QUESTIONS = df["question"].to_list()
GROUND_TRUTH = df["correct_answer"].tolist()

for model in EVAL_EMBEDDING_MODELS:
    data = {"question": [], "ground_truth": [], "contexts": []}
    data["question"] = QUESTIONS
    data["ground_truth"] = GROUND_TRUTH

    retriever = get_retriever(model, 2)
    for question in tqdm(QUESTIONS):
        data["contexts"].append(
            [doc.page_content for doc in retriever.get_relevant_documents(question)]
        )
    dataset = Dataset.from_dict(data)
    run_config = RunConfig(max_workers=4, max_wait=180)
    result = evaluate(
        dataset=dataset,
        metrics=[context_precision, context_recall],
        run_config=run_config,
        raise_exceptions=False,
    )
    logger.debug(f"Result for the {model} model: {result}")

"""
## Step 7: Compare completion models for generation
"""
logger.info("## Step 7: Compare completion models for generation")


def get_rag_chain(retriever: VectorStoreRetriever, model: str) -> RunnableSequence:
    """
    Create a basic RAG chain

    Args:
        retriever (VectorStoreRetriever): Vector store retriever object
        model (str): Chat completion model to use

    Returns:
        RunnableSequence: A RAG chain
    """
    retrieve = {
        "context": retriever
        | (lambda docs: "\n\n".join([d.page_content for d in docs])),
        "question": RunnablePassthrough(),
    }
    template = """Answer the question based only on the following context: \
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOllama(model="llama3.2")
    parse_output = StrOutputParser()

    rag_chain = retrieve | prompt | llm | parse_output
    return rag_chain

for model in ["gpt-3.5-turbo-1106", "gpt-3.5-turbo"]:
    data = {"question": [], "ground_truth": [], "contexts": [], "answer": []}
    data["question"] = QUESTIONS
    data["ground_truth"] = GROUND_TRUTH
    retriever = get_retriever("mxbai-embed-large", 2)
    rag_chain = get_rag_chain(retriever, model)
    for question in tqdm(QUESTIONS):
        data["answer"].append(rag_chain.invoke(question))
        data["contexts"].append(
            [doc.page_content for doc in retriever.get_relevant_documents(question)]
        )
    dataset = Dataset.from_dict(data)
    run_config = RunConfig(max_workers=4, max_wait=180)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        run_config=run_config,
        raise_exceptions=False,
    )
    logger.debug(f"Result for the {model} model: {result}")

"""
## Step 8: Measure overall performance of the RAG application
"""
logger.info("## Step 8: Measure overall performance of the RAG application")


data = {"question": [], "ground_truth": [], "answer": []}
data["question"] = QUESTIONS
data["ground_truth"] = GROUND_TRUTH
retriever = get_retriever("mxbai-embed-large", 2)
rag_chain = get_rag_chain(retriever, "gpt-3.5-turbo")
for question in tqdm(QUESTIONS):
    data["answer"].append(rag_chain.invoke(question))

dataset = Dataset.from_dict(data)
run_config = RunConfig(max_workers=4, max_wait=180)
result = evaluate(
    dataset=dataset,
    metrics=[answer_similarity, answer_correctness],
    run_config=run_config,
    raise_exceptions=False,
)
logger.debug(f"Overall metrics: {result}")

result_df = result.to_pandas()

result_df.head(5)

result_df[result_df["answer_correctness"] < 0.7]


plt.figure(figsize=(10, 8))
sns.heatmap(
    result_df[1:10].set_index("question")[["answer_similarity", "answer_correctness"]],
    annot=True,
    cmap="flare",
)
plt.show()

"""
## Step 9: Tracking performance over time
"""
logger.info("## Step 9: Tracking performance over time")


result["timestamp"] = datetime.now()

collection = db["metrics"]
collection.insert_one(result)

logger.info("\n\n[DONE]", bright=True)