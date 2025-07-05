import asyncio
from jet.transformers.formatters import format_json
from dotenv import load_dotenv
from evaluation.evalute_rag import *
from helper_functions import *
from helper_functions import encode_pdf, encode_from_string
from jet.llm.ollama.base_langchain import ChatOllama
from jet.logger import CustomLogger
from langchain.chains.summarize.chain import load_summarize_chain
from langchain.docstore.document import Document
import asyncio
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

"""
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/hierarchical_indices.ipynb)

# Hierarchical Indices in Document Retrieval

## Overview

This code implements a Hierarchical Indexing system for document retrieval, utilizing two levels of encoding: document-level summaries and detailed chunks. This approach aims to improve the efficiency and relevance of information retrieval by first identifying relevant document sections through summaries, then drilling down to specific details within those sections.

## Motivation

Traditional flat indexing methods can struggle with large documents or corpus, potentially missing context or returning irrelevant information. Hierarchical indexing addresses this by creating a two-tier search system, allowing for more efficient and context-aware retrieval.

## Key Components

1. PDF processing and text chunking
2. Asynchronous document summarization using Ollama's GPT-4
3. Vector store creation for both summaries and detailed chunks using FAISS and Ollama embeddings
4. Custom hierarchical retrieval function

## Method Details

### Document Preprocessing and Encoding

1. The PDF is loaded and split into documents (likely by page).
2. Each document is summarized asynchronously using GPT-4.
3. The original documents are also split into smaller, detailed chunks.
4. Two separate vector stores are created:
   - One for document-level summaries
   - One for detailed chunks

### Asynchronous Processing and Rate Limiting

1. The code uses asynchronous programming (asyncio) to improve efficiency.
2. Implements batching and exponential backoff to handle API rate limits.

### Hierarchical Retrieval

The `retrieve_hierarchical` function implements the two-tier search:

1. It first searches the summary vector store to identify relevant document sections.
2. For each relevant summary, it then searches the detailed chunk vector store, filtering by the corresponding page number.
3. This approach ensures that detailed information is retrieved only from the most relevant document sections.

## Benefits of this Approach

1. Improved Retrieval Efficiency: By first searching summaries, the system can quickly identify relevant document sections without processing all detailed chunks.
2. Better Context Preservation: The hierarchical approach helps maintain the broader context of retrieved information.
3. Scalability: This method is particularly beneficial for large documents or corpus, where flat searching might be inefficient or miss important context.
4. Flexibility: The system allows for adjusting the number of summaries and chunks retrieved, enabling fine-tuning for different use cases.

## Implementation Details

1. Asynchronous Programming: Utilizes Python's asyncio for efficient I/O operations and API calls.
2. Rate Limit Handling: Implements batching and exponential backoff to manage API rate limits effectively.
3. Persistent Storage: Saves the generated vector stores locally to avoid unnecessary recomputation.

## Conclusion

Hierarchical indexing represents a sophisticated approach to document retrieval, particularly suitable for large or complex document sets. By leveraging both high-level summaries and detailed chunks, it offers a balance between broad context understanding and specific information retrieval. This method has potential applications in various fields requiring efficient and context-aware information retrieval, such as legal document analysis, academic research, or large-scale content management systems.

<div style="text-align: center;">

<img src="../images/hierarchical_indices.svg" alt="hierarchical_indices" style="width:50%; height:auto;">
</div>

<div style="text-align: center;">

<img src="../images/hierarchical_indices_example.svg" alt="hierarchical_indices" style="width:100%; height:auto;">
</div>

# Package Installation and Imports

The cell below installs all necessary packages required to run this notebook.
"""
logger.info("# Hierarchical Indices in Document Retrieval")

# !pip install langchain langchain-openai python-dotenv

# !git clone https://github.com/NirDiamant/RAG_TECHNIQUES.git
sys.path.append('RAG_TECHNIQUES')



load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

"""
### Define document path
"""
logger.info("### Define document path")

os.makedirs('data', exist_ok=True)

# !wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
# !wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf

path = f"{GENERATED_DIR}/Understanding_Climate_Change.pdf"

"""
### Function to encode to both summary and chunk levels, sharing the page metadata
"""
logger.info("### Function to encode to both summary and chunk levels, sharing the page metadata")

async def encode_pdf_hierarchical(path, chunk_size=1000, chunk_overlap=200, is_string=False):
    """
    Asynchronously encodes a PDF book into a hierarchical vector store using Ollama embeddings.
    Includes rate limit handling with exponential backoff.

    Args:
        path: The path to the PDF file.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.

    Returns:
        A tuple containing two FAISS vector stores:
        1. Document-level summaries
        2. Detailed chunks
    """

    if not is_string:
        loader = PyPDFLoader(path)
        async def run_async_code_2eb41ac0():
            async def run_async_code_64d57633():
                documents = await asyncio.to_thread(loader.load)
                return documents
            documents = asyncio.run(run_async_code_64d57633())
            logger.success(format_json(documents))
            return documents
        documents = asyncio.run(run_async_code_2eb41ac0())
        logger.success(format_json(documents))
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        documents = text_splitter.create_documents([path])


    summary_llm = ChatOllama(model="llama3.1")
    summary_chain = load_summarize_chain(summary_llm, chain_type="map_reduce")

    async def summarize_doc(doc):
        """
        Summarizes a single document with rate limit handling.

        Args:
            doc: The document to be summarized.

        Returns:
            A summarized Document object.
        """
        async def run_async_code_9c59c0b4():
            async def run_async_code_3c173627():
                summary_output = await retry_with_exponential_backoff(summary_chain.ainvoke([doc]))
                return summary_output
            summary_output = asyncio.run(run_async_code_3c173627())
            logger.success(format_json(summary_output))
            return summary_output
        summary_output = asyncio.run(run_async_code_9c59c0b4())
        logger.success(format_json(summary_output))
        summary = summary_output['output_text']
        return Document(
            page_content=summary,
            metadata={"source": path, "page": doc.metadata["page"], "summary": True}
        )

    batch_size = 5  # Adjust this based on your rate limits
    summaries = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        async def run_async_code_bac78b97():
            async def run_async_code_2db24c6e():
                batch_summaries = await asyncio.gather(*[summarize_doc(doc) for doc in batch])
                return batch_summaries
            batch_summaries = asyncio.run(run_async_code_2db24c6e())
            logger.success(format_json(batch_summaries))
            return batch_summaries
        batch_summaries = asyncio.run(run_async_code_bac78b97())
        logger.success(format_json(batch_summaries))
        summaries.extend(batch_summaries)
        async def run_async_code_d7f88492():
            await asyncio.sleep(1)  # Short pause between batches
            return 
         = asyncio.run(run_async_code_d7f88492())
        logger.success(format_json())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    async def run_async_code_7551f81f():
        async def run_async_code_28a76fbf():
            detailed_chunks = await asyncio.to_thread(text_splitter.split_documents, documents)
            return detailed_chunks
        detailed_chunks = asyncio.run(run_async_code_28a76fbf())
        logger.success(format_json(detailed_chunks))
        return detailed_chunks
    detailed_chunks = asyncio.run(run_async_code_7551f81f())
    logger.success(format_json(detailed_chunks))

    for i, chunk in enumerate(detailed_chunks):
        chunk.metadata.update({
            "chunk_id": i,
            "summary": False,
            "page": int(chunk.metadata.get("page", 0))
        })

    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    async def create_vectorstore(docs):
        """
        Creates a vector store from a list of documents with rate limit handling.

        Args:
            docs: The list of documents to be embedded.

        Returns:
            A FAISS vector store containing the embedded documents.
        """
        return await retry_with_exponential_backoff(
            asyncio.to_thread(FAISS.from_documents, docs, embeddings)
        )

    async def async_func_85():
        summary_vectorstore, detailed_vectorstore = await asyncio.gather(
            create_vectorstore(summaries),
            create_vectorstore(detailed_chunks)
        )
        return summary_vectorstore, detailed_vectorstore
    summary_vectorstore, detailed_vectorstore = asyncio.run(async_func_85())
    logger.success(format_json(summary_vectorstore, detailed_vectorstore))

    return summary_vectorstore, detailed_vectorstore

"""
### Encode the PDF book to both document-level summaries and detailed chunks if the vector stores do not exist
"""
logger.info("### Encode the PDF book to both document-level summaries and detailed chunks if the vector stores do not exist")

if os.path.exists("../vector_stores/summary_store") and os.path.exists("../vector_stores/detailed_store"):
   embeddings = OllamaEmbeddings(model="mxbai-embed-large")
   summary_store = FAISS.load_local("../vector_stores/summary_store", embeddings, allow_dangerous_deserialization=True)
   detailed_store = FAISS.load_local("../vector_stores/detailed_store", embeddings, allow_dangerous_deserialization=True)

else:
    async def run_async_code_2253c620():
        async def run_async_code_17fecf0f():
            summary_store, detailed_store = await encode_pdf_hierarchical(path)
            return summary_store, detailed_store
        summary_store, detailed_store = asyncio.run(run_async_code_17fecf0f())
        logger.success(format_json(summary_store, detailed_store))
        return summary_store, detailed_store
    summary_store, detailed_store = asyncio.run(run_async_code_2253c620())
    logger.success(format_json(summary_store, detailed_store))
    summary_store.save_local("../vector_stores/summary_store")
    detailed_store.save_local("../vector_stores/detailed_store")

"""
### Retrieve information according to summary level, and then retrieve information from the chunk level vector store and filter according to the summary level pages
"""
logger.info("### Retrieve information according to summary level, and then retrieve information from the chunk level vector store and filter according to the summary level pages")

def retrieve_hierarchical(query, summary_vectorstore, detailed_vectorstore, k_summaries=3, k_chunks=5):
    """
    Performs a hierarchical retrieval using the query.

    Args:
        query: The search query.
        summary_vectorstore: The vector store containing document summaries.
        detailed_vectorstore: The vector store containing detailed chunks.
        k_summaries: The number of top summaries to retrieve.
        k_chunks: The number of detailed chunks to retrieve per summary.

    Returns:
        A list of relevant detailed chunks.
    """

    top_summaries = summary_vectorstore.similarity_search(query, k=k_summaries)

    relevant_chunks = []
    for summary in top_summaries:
        page_number = summary.metadata["page"]
        page_filter = lambda metadata: metadata["page"] == page_number
        page_chunks = detailed_vectorstore.similarity_search(
            query,
            k=k_chunks,
            filter=page_filter
        )
        relevant_chunks.extend(page_chunks)

    return relevant_chunks

"""
### Demonstrate on a use case
"""
logger.info("### Demonstrate on a use case")

query = "What is the greenhouse effect?"
results = retrieve_hierarchical(query, summary_store, detailed_store)

for chunk in results:
    logger.debug(f"Page: {chunk.metadata['page']}")
    logger.debug(f"Content: {chunk.page_content}...")  # Print first 100 characters
    logger.debug("---")

logger.info("\n\n[DONE]", bright=True)