from dotenv import load_dotenv
from jet.llm.ollama.base_langchain import ChatOllama,OllamaEmbeddings
from jet.logger import CustomLogger
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
import faiss
import libries
import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/simple_csv_rag.ipynb)

# Simple RAG (Retrieval-Augmented Generation) System for CSV Files

## Overview

This code implements a basic Retrieval-Augmented Generation (RAG) system for processing and querying CSV documents. The system encodes the document content into a vector store, which can then be queried to retrieve relevant information.

# CSV File Structure and Use Case
The CSV file contains dummy customer data, comprising various attributes like first name, last name, company, etc. This dataset will be utilized for a RAG use case, facilitating the creation of a customer information Q&A system.

## Key Components

1. Loading and spliting csv files.
2. Vector store creation using [FAISS](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) and Ollama embeddings
3. Retriever setup for querying the processed documents
4. Creating a question and answer over the csv data.

## Method Details

### Document Preprocessing

1. The csv is loaded using langchain Csvloader
2. The data is split into chunks.


### Vector Store Creation

1. Ollama embeddings are used to create vector representations of the text chunks.
2. A FAISS vector store is created from these embeddings for efficient similarity search.

### Retriever Setup

1. A retriever is configured to fetch the most relevant chunks for a given query.

## Benefits of this Approach

1. Scalability: Can handle large documents by processing them in chunks.
2. Flexibility: Easy to adjust parameters like chunk size and number of retrieved results.
3. Efficiency: Utilizes FAISS for fast similarity search in high-dimensional spaces.
4. Integration with Advanced NLP: Uses Ollama embeddings for state-of-the-art text representation.

## Conclusion

This simple RAG system provides a solid foundation for building more complex information retrieval and question-answering systems. By encoding document content into a searchable vector store, it enables efficient retrieval of relevant information in response to queries. This approach is particularly useful for applications requiring quick access to specific information within a csv file.


# Package Installation and Imports

The cell below installs all necessary packages required to run this notebook.
"""
logger.info("# Simple RAG (Retrieval-Augmented Generation) System for CSV Files")

# !pip install faiss-cpu langchain langchain-community langchain-openai pandas python-dotenv


load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

llm = ChatOllama(model="llama3.1")

"""
# CSV File Structure and Use Case
The CSV file contains dummy customer data, comprising various attributes like first name, last name, company, etc. This dataset will be utilized for a RAG use case, facilitating the creation of a customer information Q&A system.
"""
logger.info("# CSV File Structure and Use Case")

os.makedirs('data', exist_ok=True)

# !wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
# !wget -O data/customers-100.csv https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/customers-100.csv


file_path = ('data/customers-100.csv') # insert the path of the csv file
data = pd.read_csv(file_path)

data.head()

"""
load and process csv data
"""
logger.info("load and process csv data")

loader = CSVLoader(file_path=file_path)
docs = loader.load_and_split()

"""
Initiate faiss vector store and openai embedding
"""
logger.info("Initiate faiss vector store and openai embedding")


embeddings = OllamaEmbeddings(model="mxbai-embed-large")
index = faiss.IndexFlatL2(len(OllamaEmbeddings(model="mxbai-embed-large").embed_query(" ")))
vector_store = FAISS(
    embedding_function=OllamaEmbeddings(model="mxbai-embed-large"),
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

"""
Add the splitted csv data to the vector store
"""
logger.info("Add the splitted csv data to the vector store")

vector_store.add_documents(documents=docs)

"""
Create the retrieval chain
"""
logger.info("Create the retrieval chain")


retriever = vector_store.as_retriever()

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),

])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

"""
Query the rag bot with a question based on the CSV data
"""
logger.info("Query the rag bot with a question based on the CSV data")

answer= rag_chain.invoke({"input": "which company does sheryl Baxter work for?"})
answer['answer']

logger.info("\n\n[DONE]", bright=True)