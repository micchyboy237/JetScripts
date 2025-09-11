from datasets import load_dataset
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.llm.ollama.base_langchain import OllamaEmbeddings
from jet.logger import CustomLogger
from langchain_core.globals import set_llm_cache
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.cache import MongoDBAtlasSemanticCache
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from pymongo import MongoClient
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
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/rag/mongodb-langchain-cache-memory.ipynb)

[![View Article](https://img.shields.io/badge/View%20Article-blue)](https://www.mongodb.com/developer/products/atlas/advanced-rag-langchain-mongodb/)

# Adding Semantic Caching and Memory to your RAG Application using MongoDB and LangChain

In this notebook, we will see how to use the new MongoDBCache and MongoDBChatMessageHistory in your RAG application.

## Step 1: Install required libraries

- **datasets**: Python library to get access to datasets available on Hugging Face Hub

- **langchain**: Python toolkit for LangChain

- **langchain-mongodb**: Python package to use MongoDB as a vector store, semantic cache, chat history store etc. in LangChain

- **langchain-ollama**: Python package to use Ollama models with LangChain

- **pymongo**: Python toolkit for MongoDB

- **pandas**: Python library for data analysis, exploration, and manipulation
"""
logger.info(
    "# Adding Semantic Caching and Memory to your RAG Application using MongoDB and LangChain")

# ! pip install -qU datasets langchain langchain-mongodb langchain-ollama pymongo pandas

"""
## Step 2: Setup pre-requisites

* Set the MongoDB connection string. Follow the steps [here](https://www.mongodb.com/docs/manual/reference/connection-string/) to get the connection string from the Atlas UI.

* Set the Ollama API key. Steps to obtain an API key as [here](https://help.ollama.com/en/articles/4936850-where-do-i-find-my-ollama-api-key)
"""
logger.info("## Step 2: Setup pre-requisites")

# import getpass

# MONGODB_URI = getpass.getpass("Enter your MongoDB connection string:")

# OPENAI_API_KEY = getpass.getpass("Enter your Ollama API key:")


os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()

"""
## Step 3: Download the dataset

We will be using MongoDB's [embedded_movies](https://huggingface.co/datasets/MongoDB/embedded_movies) dataset
"""
logger.info("## Step 3: Download the dataset")


data = load_dataset("MongoDB/embedded_movies")

df = pd.DataFrame(data["train"])

"""
## Step 4: Data analysis

Make sure length of the dataset is what we expect, drop Nones etc.
"""
logger.info("## Step 4: Data analysis")

df.head(1)

df = df[df["fullplot"].notna()]

df.rename(columns={"plot_embedding": "embedding"}, inplace=True)

"""
## Step 5: Create a simple RAG chain using MongoDB as the vector store
"""
logger.info(
    "## Step 5: Create a simple RAG chain using MongoDB as the vector store")


client = MongoClient(
    MONGODB_URI, appname="devrel.showcase.mongodb_langchain_cache_memory"
)

DB_NAME = "langchain_chatbot"
COLLECTION_NAME = "data"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
collection = client[DB_NAME][COLLECTION_NAME]

collection.delete_many({})

records = df.to_dict("records")
collection.insert_many(records)

logger.debug("Data ingestion into MongoDB completed")


embeddings = OllamaEmbeddings(
    #     ollama_api_key=OPENAI_API_KEY, model="text-embedding-ada-002"
)

vector_store = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=MONGODB_URI,
    namespace=DB_NAME + "." + COLLECTION_NAME,
    embedding=embeddings,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    text_key="fullplot",
)

retriever = vector_store.as_retriever(
    search_type="similarity", search_kwargs={"k": 5})


retrieve = {
    "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
    "question": RunnablePassthrough(),
}
template = """Answer the question based only on the following context: \
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
# model = ChatOllama(model="llama3.2")
parse_output = StrOutputParser()

naive_rag_chain = retrieve | prompt | model | parse_output

naive_rag_chain.invoke("What is the best movie to watch when sad?")

"""
## Step 6: Create a RAG chain with chat history
"""
logger.info("## Step 6: Create a RAG chain with chat history")


def get_session_history(session_id: str) -> MongoDBChatMessageHistory:
    return MongoDBChatMessageHistory(
        MONGODB_URI, session_id, database_name=DB_NAME, collection_name="history"
    )


standalone_system_prompt = """
Given a chat history and a follow-up question, rephrase the follow-up question to be a standalone question. \
Do NOT answer the question, just reformulate it if needed, otherwise return it as is. \
Only return the final standalone question. \
"""
standalone_question_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", standalone_system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

question_chain = standalone_question_prompt | model | parse_output

retriever_chain = RunnablePassthrough.assign(
    context=question_chain
    | retriever
    | (lambda docs: "\n\n".join([d.page_content for d in docs]))
)

rag_system_prompt = """Answer the question based only on the following context: \
{context}
"""
rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rag_system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

rag_chain = retriever_chain | rag_prompt | model | parse_output

with_message_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)
with_message_history.invoke(
    {"question": "What is the best movie to watch when sad?"},
    {"configurable": {"session_id": "1"}},
)

with_message_history.invoke(
    {
        "question": "Hmmm..I don't want to watch that one. Can you suggest something else?"
    },
    {"configurable": {"session_id": "1"}},
)

with_message_history.invoke(
    {"question": "How about something more light?"},
    {"configurable": {"session_id": "1"}},
)

"""
## Step 7: Get faster responses using Semantic Cache

**NOTE:** Semantic cache only caches the input to the LLM. When using it in retrieval chains, remember that documents retrieved can change between runs resulting in cache misses for semantically similar queries.
"""
logger.info("## Step 7: Get faster responses using Semantic Cache")


set_llm_cache(
    MongoDBAtlasSemanticCache(
        connection_string=MONGODB_URI,
        embedding=embeddings,
        collection_name="semantic_cache",
        database_name=DB_NAME,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
        wait_until_ready=True,  # Optional, waits until the cache is ready to be used
    )
)

# %%time
naive_rag_chain.invoke("What is the best movie to watch when sad?")

# %%time
naive_rag_chain.invoke("What is the best movie to watch when sad?")

# %%time
naive_rag_chain.invoke("Which movie do I watch when sad?")

logger.info("\n\n[DONE]", bright=True)
