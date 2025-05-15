from dotenv import load_dotenv
from evaluation.evalute_rag import *
from helper_functions import *
from jet.logger import CustomLogger
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join(script_dir, "generated", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

"""
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/contextual_compression.ipynb)

# Contextual Compression in Document Retrieval

## Overview

This code demonstrates the implementation of contextual compression in a document retrieval system using LangChain and Ollama's language models. The technique aims to improve the relevance and conciseness of retrieved information by compressing and extracting the most pertinent parts of documents in the context of a given query.

## Motivation

Traditional document retrieval systems often return entire chunks or documents, which may contain irrelevant information. Contextual compression addresses this by intelligently extracting and compressing only the most relevant parts of retrieved documents, leading to more focused and efficient information retrieval.

## Key Components

1. Vector store creation from a PDF document
2. Base retriever setup
3. LLM-based contextual compressor
4. Contextual compression retriever
5. Question-answering chain integrating the compressed retriever

## Method Details

### Document Preprocessing and Vector Store Creation

1. The PDF is processed and encoded into a vector store using a custom `encode_pdf` function.

### Retriever and Compressor Setup

1. A base retriever is created from the vector store.
2. An LLM-based contextual compressor (LLMChainExtractor) is initialized using Ollama's GPT-4 model.

### Contextual Compression Retriever

1. The base retriever and compressor are combined into a ContextualCompressionRetriever.
2. This retriever first fetches documents using the base retriever, then applies the compressor to extract the most relevant information.

### Question-Answering Chain

1. A RetrievalQA chain is created, integrating the compression retriever.
2. This chain uses the compressed and extracted information to generate answers to queries.

## Benefits of this Approach

1. Improved relevance: The system returns only the most pertinent information to the query.
2. Increased efficiency: By compressing and extracting relevant parts, it reduces the amount of text the LLM needs to process.
3. Enhanced context understanding: The LLM-based compressor can understand the context of the query and extract information accordingly.
4. Flexibility: The system can be easily adapted to different types of documents and queries.

## Conclusion

Contextual compression in document retrieval offers a powerful way to enhance the quality and efficiency of information retrieval systems. By intelligently extracting and compressing relevant information, it provides more focused and context-aware responses to queries. This approach has potential applications in various fields requiring efficient and accurate information retrieval from large document collections.

<div style="text-align: center;">

<img src="../images/contextual_compression.svg" alt="contextual compression" style="width:70%; height:auto;">
</div>

# Package Installation and Imports

The cell below installs all necessary packages required to run this notebook.
"""
logger.info("# Contextual Compression in Document Retrieval")

# !pip install langchain python-dotenv

# !git clone https://github.com/N7/RAG_TECHNIQUES.git
sys.path.append('RAG_TECHNIQUES')


load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

"""
### Define document's path
"""
logger.info("### Define document's path")

os.makedirs('data', exist_ok=True)

# !wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/N7/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
# !wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/N7/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf

path = f"{GENERATED_DIR}/Understanding_Climate_Change.pdf"

"""
### Create a vector store
"""
logger.info("### Create a vector store")

vector_store = encode_pdf(path)

"""
### Create a retriever + contexual compressor + combine them
"""
logger.info("### Create a retriever + contexual compressor + combine them")

retriever = vector_store.as_retriever()


llm = ChatOllama(model="llama3.1")
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=compression_retriever,
    return_source_documents=True
)

"""
### Example usage
"""
logger.info("### Example usage")

query = "What is the main topic of the document?"
result = qa_chain.invoke({"query": query})
logger.debug(result["result"])
logger.debug("Source documents:", result["source_documents"])

logger.info("\n\n[DONE]", bright=True)
