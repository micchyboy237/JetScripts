[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/contextual_compression.ipynb)

# Contextual Compression in Document Retrieval

## Overview

This code demonstrates the implementation of contextual compression in a document retrieval system using LangChain and OpenAI's language models. The technique aims to improve the relevance and conciseness of retrieved information by compressing and extracting the most pertinent parts of documents in the context of a given query.

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
2. An LLM-based contextual compressor (LLMChainExtractor) is initialized using OpenAI's GPT-4 model.

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

```python
# Install required packages
!pip install langchain python-dotenv
```

```python
# Clone the repository to access helper functions and evaluation modules
!git clone https://github.com/N7/RAG_TECHNIQUES.git
import sys
sys.path.append('RAG_TECHNIQUES')
# If you need to run with the latest data
# !cp -r RAG_TECHNIQUES/data .
```

```python
import os
import sys
from dotenv import load_dotenv
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains import RetrievalQA


# Original path append replaced for Colab compatibility
from helper_functions import *
from evaluation.evalute_rag import *

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
```

### Define document's path

```python
# Download required data files
import os
os.makedirs('data', exist_ok=True)

# Download the PDF document used in this notebook
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/N7/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/N7/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
```

```python
path = "data/Understanding_Climate_Change.pdf"
```

### Create a vector store

```python
vector_store = encode_pdf(path)
```

### Create a retriever + contexual compressor + combine them

```python
# Create a retriever
retriever = vector_store.as_retriever()


#Create a contextual compressor
llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=4000)
compressor = LLMChainExtractor.from_llm(llm)

#Combine the retriever with the compressor
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

# Create a QA chain with the compressed retriever
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=compression_retriever,
    return_source_documents=True
)
```

### Example usage

```python
query = "What is the main topic of the document?"
result = qa_chain.invoke({"query": query})
print(result["result"])
print("Source documents:", result["source_documents"])
```
