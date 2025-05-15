from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from evaluation.evalute_rag import *
from helper_functions import *
from jet.logger import CustomLogger
from langchain_community.docstore.in_memory import InMemoryDocstore
from tqdm import tqdm
import faiss
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
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/HyPE_Hypothetical_Prompt_Embeddings.ipynb)

# Hypothetical Prompt Embeddings (HyPE)

## Overview

This code implements a Retrieval-Augmented Generation (RAG) system enhanced by Hypothetical Prompt Embeddings (HyPE). Unlike traditional RAG pipelines that struggle with query-document style mismatch, HyPE precomputes hypothetical questions during the indexing phase. This transforms retrieval into a question-question matching problem, eliminating the need for expensive runtime query expansion techniques.

## Key Components of notebook

1. PDF processing and text extraction
2. Text chunking to maintain coherent information units
3. **Hypothetical Prompt Embedding Generation** using an LLM to create multiple proxy questions per chunk
4. Vector store creation using [FAISS](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) and Ollama embeddings
5. Retriever setup for querying the processed documents
6. Evaluation of the RAG system

## Method Details

### Document Preprocessing

1. The PDF is loaded using `PyPDFLoader`.
2. The text is split into chunks using `RecursiveCharacterTextSplitter` with specified chunk size and overlap.

### Hypothetical Question Generation

Instead of embedding raw text chunks, HyPE **generates multiple hypothetical prompts** for each chunk. These **precomputed questions** simulate user queries, improving alignment with real-world searches. This removes the need for runtime synthetic answer generation needed in techniques like HyDE.

### Vector Store Creation

1. Each hypothetical question is embedded using Ollama embeddings.
2. A FAISS vector store is built, associating **each question embedding with its original chunk**.
3. This approach **stores multiple representations per chunk**, increasing retrieval flexibility.

### Retriever Setup

1. The retriever is optimized for **question-question matching** rather than direct document retrieval.
2. The FAISS index enables **efficient nearest-neighbor** search over the hypothetical prompt embeddings.
3. Retrieved chunks provide a **richer and more precise context** for downstream LLM generation.

## Key Features

1. **Precomputed Hypothetical Prompts** – Improves query alignment without runtime overhead.
2. **Multi-Vector Representation**– Each chunk is indexed multiple times for broader semantic coverage.
3. **Efficient Retrieval** – FAISS ensures fast similarity search over the enhanced embeddings.
4. **Modular Design** – The pipeline is easy to adapt for different datasets and retrieval settings. Additionally it's compatible with most optimizations like reranking etc.

## Evaluation

HyPE's effectiveness is evaluated across multiple datasets, showing:

- Up to 42 percentage points improvement in retrieval precision
- Up to 45 percentage points improvement in claim recall
    (See full evaluation results in [preprint](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5139335))

## Benefits of this Approach

1. **Eliminates Query-Time Overhead** – All hypothetical generation is done offline at indexing.
2. **Enhanced Retrieval Precision** – Better alignment between queries and stored content.
3. **Scalable & Efficient** – No addinal per-query computational cost; retrieval is as fast as standard RAG.
4. **Flexible & Extensible** – Can be combined with advanced RAG techniques like reranking.

## Conclusion

HyPE provides a scalable and efficient alternative to traditional RAG systems, overcoming query-document style mismatch while avoiding the computational cost of runtime query expansion. By moving hypothetical prompt generation to indexing, it significantly enhances retrieval precision and efficiency, making it a practical solution for real-world applications.

For further details, refer to the full paper: [preprint](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5139335)


<div style="text-align: center;">

<img src="../images/hype.svg" alt="HyPE" style="width:70%; height:auto;">
</div>

# Package Installation and Imports

The cell below installs all necessary packages required to run this notebook.
"""
logger.info("# Hypothetical Prompt Embeddings (HyPE)")

# !pip install faiss-cpu futures langchain-community python-dotenv tqdm

# !git clone https://github.com/N7/RAG_TECHNIQUES.git
sys.path.append('RAG_TECHNIQUES')


load_dotenv()

# if not os.getenv('OPENAI_API_KEY'):
#     os.environ["OPENAI_API_KEY"] = input("Please enter your Ollama API key: ")
else:
    #     os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


"""
### Define constants

- `PATH`: path to the data, to be embedded into the RAG pipeline

This tutorial uses Ollama endpoint ([avalible models](https://platform.openai.com/docs/pricing)). 
- `LANGUAGE_MODEL_NAME`: The name of the language model to be used. 
- `EMBEDDING_MODEL_NAME`: The name of the embedding model to be used.

The tutroial uses a `RecursiveCharacterTextSplitter` chunking approach where the chunking length function used is python `len` function. The chunking varables to be tweaked here are:
- `CHUNK_SIZE`: The minimum length of one chunk
- `CHUNK_OVERLAP`: The overlap of two consecutive chunks.
"""
logger.info("### Define constants")

os.makedirs('data', exist_ok=True)

# !wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/N7/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
# !wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/N7/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf

PATH = f"{GENERATED_DIR}/Understanding_Climate_Change.pdf"
LANGUAGE_MODEL_NAME = "llama3.1"
EMBEDDING_MODEL_NAME = "mxbai-embed-large"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

"""
### Define generation of Hypothetical Prompt Embeddings

The code block below generates hypothetical questions for each text chunk and embeds them for retrieval.

- An LLM extracts key questions from the input chunk.
- These questions are embedded using Ollama's model.
- The function returns the original chunk and its prompt embeddings later used for retrieval.

To ensure clean output, extra newlines are removed, and regex parsing can improve list formatting when needed.
"""
logger.info("### Define generation of Hypothetical Prompt Embeddings")


def generate_hypothetical_prompt_embeddings(chunk_text: str):
    """
    Uses the LLM to generate multiple hypothetical questions for a single chunk.
    These questions will be used as 'proxies' for the chunk during retrieval.

    Parameters:
    chunk_text (str): Text contents of the chunk

    Returns:
    chunk_text (str): Text contents of the chunk. This is done to make the
        multithreading easier
    hypothetical prompt embeddings (List[float]): A list of embedding vectors
        generated from the questions
    """
    llm = ChatOllama(model="llama3.1")
    embedding_model = OllamaEmbeddings(model="mxbai-embed-large")

    question_gen_prompt = PromptTemplate.from_template(
        "Analyze the input text and generate essential questions that, when answered, \
        capture the main points of the text. Each question should be one line, \
        without numbering or prefixes.\n\n \
        Text:\n{chunk_text}\n\nQuestions:\n"
    )
    question_chain = question_gen_prompt | llm | StrOutputParser()

    questions = question_chain.invoke(
        {"chunk_text": chunk_text}).replace("\n\n", "\n").split("\n")

    return chunk_text, embedding_model.embed_documents(questions)


"""
### Define creation and population of FAISS Vector Store

The code block below builds a FAISS vector store by embedding text chunks in parallel.

What happens?
- Parallel processing – Uses threading to generate embeddings faster.
- FAISS initialization – Sets up an L2 index for efficient similarity search.
- Chunk embedding – Each chunk is stored multiple times, once for each generated question embedding.
- In-memory storage – Uses InMemoryDocstore for fast lookup.

This ensures efficient retrieval, improving query alignment with precomputed question embeddings.
"""
logger.info("### Define creation and population of FAISS Vector Store")


def prepare_vector_store(chunks: List[str]):
    """
    Creates and populates a FAISS vector store from a list of text chunks.

    This function processes a list of text chunks in parallel, generating
    hypothetical prompt embeddings for each chunk.
    The embeddings are stored in a FAISS index for efficient similarity search.

    Parameters:
    chunks (List[str]): A list of text chunks to be embedded and stored.

    Returns:
    FAISS: A FAISS vector store containing the embedded text chunks.
    """

    vector_store = None

    with ThreadPoolExecutor() as pool:
        futures = [pool.submit(
            generate_hypothetical_prompt_embeddings, c) for c in chunks]

        for f in tqdm(as_completed(futures), total=len(chunks)):

            chunk, vectors = f.result()  # Retrieve the processed chunk and its embeddings

            if vector_store == None:
                vector_store = FAISS(
                    embedding_function=OllamaEmbeddings(
                        model="mxbai-embed-large"),  # Define embedding model
                    # Define an L2 index for similarity search
                    index=faiss.IndexFlatL2(len(vectors[0]))
                    docstore=InMemoryDocstore(),  # Use in-memory document storage
                    index_to_docstore_id={}  # Maintain index-to-document mapping
                )

            chunks_with_embedding_vectors = [
                (chunk.page_content, vec) for vec in vectors]

            vector_store.add_embeddings(chunks_with_embedding_vectors)

    return vector_store  # Return the populated vector store


"""
### Encode PDF into a FAISS Vector Store

The code block below processes a PDF file and stores its content as embeddings for retrieval.

What happens?
- PDF loading – Extracts text from the document.
- Chunking – Splits text into overlapping segments for better context retention.
- Preprocessing – Cleans text to improve embedding quality.
- Vector store creation – Generates embeddings and stores them in FAISS for retrieval.
"""
logger.info("### Encode PDF into a FAISS Vector Store")


def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    Encodes a PDF book into a vector store using Ollama embeddings.

    Args:
        path: The path to the PDF file.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.

    Returns:
        A FAISS vector store containing the encoded book content.
    """

    loader = PyPDFLoader(path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    vectorstore = prepare_vector_store(cleaned_texts)

    return vectorstore


"""
### Create HyPE vector store

Now we process the PDF and store its embeddings.
This step initializes the FAISS vector store with the encoded document.
"""
logger.info("### Create HyPE vector store")

chunks_vector_store = encode_pdf(
    PATH, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

"""
### Create retriever

Now we set up the retriever to fetch relevant chunks from the vector store.

Retrieves the top `k=3` most relevant chunks based on query similarity.
"""
logger.info("### Create retriever")

chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={
                                                          "k": 3})

"""
### Test retriever

Now we test retrieval using a sample query.

- Queries the vector store to find the most relevant chunks.
- Deduplicates results to remove potentially repeated chunks.
- Displays the retrieved context for inspection.

This step verifies that the retriever returns meaningful and diverse information for the given question.
"""
logger.info("### Test retriever")

test_query = "What is the main cause of climate change?"
context = retrieve_context_per_question(test_query, chunks_query_retriever)
context = list(set(context))
show_context(context)

"""
### Evaluate results
"""
logger.info("### Evaluate results")

evaluate_rag(chunks_query_retriever)

logger.info("\n\n[DONE]", bright=True)
