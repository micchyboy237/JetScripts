from dotenv import load_dotenv
from evaluation.evalute_rag import *
from helper_functions import *
from jet.logger import CustomLogger
from langchain.docstore.document import Document
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
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/context_enrichment_window_around_chunk.ipynb)

# Context Enrichment Window for Document Retrieval

## Overview

This code implements a context enrichment window technique for document retrieval in a vector database. It enhances the standard retrieval process by adding surrounding context to each retrieved chunk, improving the coherence and completeness of the returned information.

## Motivation

Traditional vector search often returns isolated chunks of text, which may lack necessary context for full understanding. This approach aims to provide a more comprehensive view of the retrieved information by including neighboring text chunks.

## Key Components

1. PDF processing and text chunking
2. Vector store creation using FAISS and Ollama embeddings
3. Custom retrieval function with context window
4. Comparison between standard and context-enriched retrieval

## Method Details

### Document Preprocessing

1. The PDF is read and converted to a string.
2. The text is split into chunks with overlap, each chunk tagged with its index.

### Vector Store Creation

1. Ollama embeddings are used to create vector representations of the chunks.
2. A FAISS vector store is created from these embeddings.

### Context-Enriched Retrieval

1. The `retrieve_with_context_overlap` function performs the following steps:
   - Retrieves relevant chunks based on the query
   - For each relevant chunk, fetches neighboring chunks
   - Concatenates the chunks, accounting for overlap
   - Returns the expanded context for each relevant chunk

### Retrieval Comparison

The notebook includes a section to compare standard retrieval with the context-enriched approach.

## Benefits of this Approach

1. Provides more coherent and contextually rich results
2. Maintains the advantages of vector search while mitigating its tendency to return isolated text fragments
3. Allows for flexible adjustment of the context window size

## Conclusion

This context enrichment window technique offers a promising way to improve the quality of retrieved information in vector-based document search systems. By providing surrounding context, it helps maintain the coherence and completeness of the retrieved information, potentially leading to better understanding and more accurate responses in downstream tasks such as question answering.

<div style="text-align: center;">

<img src="../images/vector-search-comparison_context_enrichment.svg" alt="context enrichment window" style="width:70%; height:auto;">
</div>

<div style="text-align: center;">

<img src="../images/context_enrichment_window.svg" alt="context enrichment window" style="width:70%; height:auto;">
</div>

# Package Installation and Imports

The cell below installs all necessary packages required to run this notebook.
"""
logger.info("# Context Enrichment Window for Document Retrieval")

# !pip install langchain python-dotenv

# !git clone https://github.com/NirDiamant/RAG_TECHNIQUES.git
sys.path.append('RAG_TECHNIQUES')




load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

"""
### Define path to PDF
"""
logger.info("### Define path to PDF")

os.makedirs('data', exist_ok=True)

# !wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
# !wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf

path = f"{GENERATED_DIR}/Understanding_Climate_Change.pdf"

"""
### Read PDF to string
"""
logger.info("### Read PDF to string")

content = read_pdf_to_string(path)

"""
### Function to split text into chunks with metadata of the chunk chronological index
"""
logger.info("### Function to split text into chunks with metadata of the chunk chronological index")

def split_text_to_chunks_with_indices(text: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(Document(page_content=chunk, metadata={"index": len(chunks), "text": text}))
        start += chunk_size - chunk_overlap
    return chunks

"""
### Split our document accordingly
"""
logger.info("### Split our document accordingly")

chunks_size = 400
chunk_overlap = 200
docs = split_text_to_chunks_with_indices(content, chunks_size, chunk_overlap)

"""
### Create vector store and retriever
"""
logger.info("### Create vector store and retriever")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore = FAISS.from_documents(docs, embeddings)
chunks_query_retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

"""
### Function to draw the k<sup>th</sup> chunk (in the original order) from the vector store
"""
logger.info("### Function to draw the k<sup>th</sup> chunk (in the original order) from the vector store")

def get_chunk_by_index(vectorstore, target_index: int) -> Document:
    """
    Retrieve a chunk from the vectorstore based on its index in the metadata.

    Args:
    vectorstore (VectorStore): The vectorstore containing the chunks.
    target_index (int): The index of the chunk to retrieve.

    Returns:
    Optional[Document]: The retrieved chunk as a Document object, or None if not found.
    """
    all_docs = vectorstore.similarity_search("", k=vectorstore.index.ntotal)
    for doc in all_docs:
        if doc.metadata.get('index') == target_index:
            return doc
    return None

"""
### Check the function
"""
logger.info("### Check the function")

chunk = get_chunk_by_index(vectorstore, 0)
logger.debug(chunk.page_content)

"""
### Function that retrieves from the vector stroe based on semantic similarity and then pads each retrieved chunk with its num_neighbors before and after, taking into account the chunk overlap to construct a meaningful wide window arround it
"""
logger.info("### Function that retrieves from the vector stroe based on semantic similarity and then pads each retrieved chunk with its num_neighbors before and after, taking into account the chunk overlap to construct a meaningful wide window arround it")

def retrieve_with_context_overlap(vectorstore, retriever, query: str, num_neighbors: int = 1, chunk_size: int = 200, chunk_overlap: int = 20) -> List[str]:
    """
    Retrieve chunks based on a query, then fetch neighboring chunks and concatenate them,
    accounting for overlap and correct indexing.

    Args:
    vectorstore (VectorStore): The vectorstore containing the chunks.
    retriever: The retriever object to get relevant documents.
    query (str): The query to search for relevant chunks.
    num_neighbors (int): The number of chunks to retrieve before and after each relevant chunk.
    chunk_size (int): The size of each chunk when originally split.
    chunk_overlap (int): The overlap between chunks when originally split.

    Returns:
    List[str]: List of concatenated chunk sequences, each centered on a relevant chunk.
    """
    relevant_chunks = retriever.get_relevant_documents(query)
    result_sequences = []

    for chunk in relevant_chunks:
        current_index = chunk.metadata.get('index')
        if current_index is None:
            continue

        start_index = max(0, current_index - num_neighbors)
        end_index = current_index + num_neighbors + 1  # +1 because range is exclusive at the end

        neighbor_chunks = []
        for i in range(start_index, end_index):
            neighbor_chunk = get_chunk_by_index(vectorstore, i)
            if neighbor_chunk:
                neighbor_chunks.append(neighbor_chunk)

        neighbor_chunks.sort(key=lambda x: x.metadata.get('index', 0))

        concatenated_text = neighbor_chunks[0].page_content
        for i in range(1, len(neighbor_chunks)):
            current_chunk = neighbor_chunks[i].page_content
            overlap_start = max(0, len(concatenated_text) - chunk_overlap)
            concatenated_text = concatenated_text[:overlap_start] + current_chunk

        result_sequences.append(concatenated_text)

    return result_sequences

"""
### Comparing regular retrival and retrival with context window
"""
logger.info("### Comparing regular retrival and retrival with context window")

query = "Explain the role of deforestation and fossil fuels in climate change."
baseline_chunk = chunks_query_retriever.get_relevant_documents(query
    ,
    k=1
)
enriched_chunks = retrieve_with_context_overlap(
    vectorstore,
    chunks_query_retriever,
    query,
    num_neighbors=1,
    chunk_size=400,
    chunk_overlap=200
)

logger.debug("Baseline Chunk:")
logger.debug(baseline_chunk[0].page_content)
logger.debug("\nEnriched Chunks:")
logger.debug(enriched_chunks[0])

"""
### An example that showcases the superiority of additional context window
"""
logger.info("### An example that showcases the superiority of additional context window")

document_content = """
Artificial Intelligence (AI) has a rich history dating back to the mid-20th century. The term "Artificial Intelligence" was coined in 1956 at the Dartmouth Conference, marking the field's official beginning.

In the 1950s and 1960s, AI research focused on symbolic methods and problem-solving. The Logic Theorist, created in 1955 by Allen Newell and Herbert A. Simon, is often considered the first AI program.

The 1960s saw the development of expert systems, which used predefined rules to solve complex problems. DENDRAL, created in 1965, was one of the first expert systems, designed to analyze chemical compounds.

However, the 1970s brought the first "AI Winter," a period of reduced funding and interest in AI research, largely due to overpromised capabilities and underdelivered results.

The 1980s saw a resurgence with the popularization of expert systems in corporations. The Japanese government's Fifth Generation Computer Project also spurred increased investment in AI research globally.

Neural networks gained prominence in the 1980s and 1990s. The backpropagation algorithm, although discovered earlier, became widely used for training multi-layer networks during this time.

The late 1990s and 2000s marked the rise of machine learning approaches. Support Vector Machines (SVMs) and Random Forests became popular for various classification and regression tasks.

Deep Learning, a subset of machine learning using neural networks with many layers, began to show promising results in the early 2010s. The breakthrough came in 2012 when a deep neural network significantly outperformed other machine learning methods in the ImageNet competition.

Since then, deep learning has revolutionized many AI applications, including image and speech recognition, natural language processing, and game playing. In 2016, Google's AlphaGo defeated a world champion Go player, a landmark achievement in AI.

The current era of AI is characterized by the integration of deep learning with other AI techniques, the development of more efficient and powerful hardware, and the ethical considerations surrounding AI deployment.

Transformers, introduced in 2017, have become a dominant architecture in natural language processing, enabling models like GPT (Generative Pre-trained Transformer) to generate human-like text.

As AI continues to evolve, new challenges and opportunities arise. Explainable AI, robust and fair machine learning, and artificial general intelligence (AGI) are among the key areas of current and future research in the field.
"""

chunks_size = 250
chunk_overlap = 20
document_chunks = split_text_to_chunks_with_indices(document_content, chunks_size, chunk_overlap)
document_vectorstore = FAISS.from_documents(document_chunks, embeddings)
document_retriever = document_vectorstore.as_retriever(search_kwargs={"k": 1})

query = "When did deep learning become prominent in AI?"
context = document_retriever.get_relevant_documents(query)
context_pages_content = [doc.page_content for doc in context]

logger.debug("Regular retrieval:\n")
show_context(context_pages_content)

sequences = retrieve_with_context_overlap(document_vectorstore, document_retriever, query, num_neighbors=1)
logger.debug("\nRetrieval with context enrichment:\n")
show_context(sequences)

logger.info("\n\n[DONE]", bright=True)