from dotenv import load_dotenv
from evaluation.evalute_rag import *
from helper_functions import *
from jet.logger import CustomLogger
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
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/HyDe_Hypothetical_Document_Embedding.ipynb)

# Hypothetical Document Embedding (HyDE) in Document Retrieval

## Overview

This code implements a Hypothetical Document Embedding (HyDE) system for document retrieval. HyDE is an innovative approach that transforms query questions into hypothetical documents containing the answer, aiming to bridge the gap between query and document distributions in vector space.

## Motivation

Traditional retrieval methods often struggle with the semantic gap between short queries and longer, more detailed documents. HyDE addresses this by expanding the query into a full hypothetical document, potentially improving retrieval relevance by making the query representation more similar to the document representations in the vector space.

## Key Components

1. PDF processing and text chunking
2. Vector store creation using FAISS and Ollama embeddings
3. Language model for generating hypothetical documents
4. Custom HyDERetriever class implementing the HyDE technique

## Method Details

### Document Preprocessing and Vector Store Creation

1. The PDF is processed and split into chunks.
2. A FAISS vector store is created using Ollama embeddings for efficient similarity search.

### Hypothetical Document Generation

1. A language model (GPT-4) is used to generate a hypothetical document that answers the given query.
2. The generation is guided by a prompt template that ensures the hypothetical document is detailed and matches the chunk size used in the vector store.

### Retrieval Process

The `HyDERetriever` class implements the following steps:

1. Generate a hypothetical document from the query using the language model.
2. Use the hypothetical document as the search query in the vector store.
3. Retrieve the most similar documents to this hypothetical document.

## Key Features

1. Query Expansion: Transforms short queries into detailed hypothetical documents.
2. Flexible Configuration: Allows adjustment of chunk size, overlap, and number of retrieved documents.
3. Integration with Ollama Models: Uses GPT-4 for hypothetical document generation and Ollama embeddings for vector representation.

## Benefits of this Approach

1. Improved Relevance: By expanding queries into full documents, HyDE can potentially capture more nuanced and relevant matches.
2. Handling Complex Queries: Particularly useful for complex or multi-faceted queries that might be difficult to match directly.
3. Adaptability: The hypothetical document generation can adapt to different types of queries and document domains.
4. Potential for Better Context Understanding: The expanded query might better capture the context and intent behind the original question.

## Implementation Details

1. Uses Ollama's ChatGPT model for hypothetical document generation.
2. Employs FAISS for efficient similarity search in the vector space.
3. Allows for easy visualization of both the hypothetical document and retrieved results.

## Conclusion

Hypothetical Document Embedding (HyDE) represents an innovative approach to document retrieval, addressing the semantic gap between queries and documents. By leveraging advanced language models to expand queries into hypothetical documents, HyDE has the potential to significantly improve retrieval relevance, especially for complex or nuanced queries. This technique could be particularly valuable in domains where understanding query intent and context is crucial, such as legal research, academic literature review, or advanced information retrieval systems.

<div style="text-align: center;">

<img src="../images/HyDe.svg" alt="HyDe" style="width:40%; height:auto;">
</div>

<div style="text-align: center;">

<img src="../images/hyde-advantages.svg" alt="HyDe" style="width:100%; height:auto;">
</div>

# Package Installation and Imports

The cell below installs all necessary packages required to run this notebook.
"""
logger.info("# Hypothetical Document Embedding (HyDE) in Document Retrieval")

# !pip install python-dotenv

# !git clone https://github.com/N7/RAG_TECHNIQUES.git
sys.path.append('RAG_TECHNIQUES')


load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

"""
### Define document(s) path
"""
logger.info("### Define document(s) path")

os.makedirs('data', exist_ok=True)

# !wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/N7/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
# !wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/N7/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf

path = f"{GENERATED_DIR}/Understanding_Climate_Change.pdf"

"""
### Define the HyDe retriever class - creating vector store, generating hypothetical document, and retrieving
"""
logger.info("### Define the HyDe retriever class - creating vector store, generating hypothetical document, and retrieving")


class HyDERetriever:
    def __init__(self, files_path, chunk_size=500, chunk_overlap=100):
        self.llm = ChatOllama(model="llama3.1")

        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore = encode_pdf(
            files_path, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

        self.hyde_prompt = PromptTemplate(
            input_variables=["query", "chunk_size"],
            template="""Given the question '{query}', generate a hypothetical document that directly answers this question. The document should be detailed and in-depth.
            the document size has be exactly {chunk_size} characters.""",
        )
        self.hyde_chain = self.hyde_prompt | self.llm

    def generate_hypothetical_document(self, query):
        input_variables = {"query": query, "chunk_size": self.chunk_size}
        return self.hyde_chain.invoke(input_variables).content

    def retrieve(self, query, k=3):
        hypothetical_doc = self.generate_hypothetical_document(query)
        similar_docs = self.vectorstore.similarity_search(
            hypothetical_doc, k=k)
        return similar_docs, hypothetical_doc


"""
### Create a HyDe retriever instance
"""
logger.info("### Create a HyDe retriever instance")

retriever = HyDERetriever(path)

"""
### Demonstrate on a use case
"""
logger.info("### Demonstrate on a use case")

test_query = "What is the main cause of climate change?"
results, hypothetical_doc = retriever.retrieve(test_query)

"""
### Plot the hypothetical document and the retrieved documnets
"""
logger.info("### Plot the hypothetical document and the retrieved documnets")

docs_content = [doc.page_content for doc in results]

logger.debug("hypothetical_doc:\n")
logger.debug(text_wrap(hypothetical_doc)+"\n")
show_context(docs_content)

logger.info("\n\n[DONE]", bright=True)
