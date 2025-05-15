from dotenv import load_dotenv
from jet.logger import CustomLogger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import Ollama
from typing import List
import cohere
import os
import tiktoken

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join(script_dir, "generated", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

"""
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/contextual_chunk_headers.ipynb)

# Contextual Chunk Headers (CCH)

## Overview

Contextual chunk headers (CCH) is a method of creating chunk headers that contain higher-level context (such as document-level or section-level context), and prepending those chunk headers to the chunks prior to embedding them. This gives the embeddings a much more accurate and complete representation of the content and meaning of the text. In our testing, this feature leads to a substantial improvement in retrieval quality. In addition to increasing the rate at which the correct information is retrieved, CCH also reduces the rate at which irrelevant results show up in the search results. This reduces the rate at which the LLM misinterprets a piece of text in downstream chat and generation applications.

## Motivation

Many of the problems developers face with RAG come down to this: Individual chunks oftentimes do not contain sufficient context to be properly used by the retrieval system or the LLM. This leads to the inability to answer questions and, more worryingly, hallucinations.

Examples of this problem
- Chunks oftentimes refer to their subject via implicit references and pronouns. This causes them to not be retrieved when they should be, or to not be properly understood by the LLM.
- Individual chunks oftentimes only make sense in the context of the entire section or document, and can be misleading when read on their own.

## Key Components

#### Contextual chunk headers
The idea here is to add in higher-level context to the chunk by prepending a chunk header. This chunk header could be as simple as just the document title, or it could use a combination of document title, a concise document summary, and the full hierarchy of section and sub-section titles.

## Method Details

#### Context generation
In the demonstration below we use an LLM to generate a descriptive title for the document. This is done through a simple prompt where you pass in a truncated version of the document text and ask the LLM to generate a descriptive title for the document. If you already have sufficiently descriptive document titles then you can directly use those instead. We've found that a document title is the simplest and most important kind of higher-level context to include in the chunk header.

Other kinds of context you can include in the chunk header:
- Concise document summary
- Section/sub-section title(s)
    - This helps the retrieval system handle queries for larger sections or topics in documents.

#### Embed chunks with chunk headers
The text you embed for each chunk is simply the concatenation of the chunk header and the chunk text. If you use a reranker during retrieval, you'll want to make sure you use this same concatenation there too.

#### Add chunk headers to search results
Including the chunk headers when presenting the search results to the LLM is also beneficial as it gives the LLM more context, and makes it less likely that it misunderstands the meaning of a chunk.

![Your Technique Name](../images/contextual_chunk_headers.svg)

## Setup

You'll need a Cohere API key and an Ollama API key for this notebook.

# Package Installation and Imports

The cell below installs all necessary packages required to run this notebook.
"""
logger.info("# Contextual Chunk Headers (CCH)")

# !pip install langchain openai python-dotenv tiktoken


load_dotenv()
os.environ["CO_API_KEY"] = os.getenv('CO_API_KEY')  # Cohere API key
# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY') # Ollama API key

"""
## Load the document and split it into chunks
We'll use the basic LangChain RecursiveCharacterTextSplitter for this demo, but you can combine CCH with more sophisticated chunking methods for even better performance.
"""
logger.info("## Load the document and split it into chunks")

os.makedirs('data', exist_ok=True)

# !wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/N7/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
# !wget -O data/nike_2023_annual_report.txt https://raw.githubusercontent.com/N7/RAG_TECHNIQUES/main/data/nike_2023_annual_report.txt


def split_into_chunks(text: str, chunk_size: int = 800) -> list[str]:
    """
    Split a given text into chunks of specified size using RecursiveCharacterTextSplitter.

    Args:
        text (str): The input text to be split into chunks.
        chunk_size (int, optional): The maximum size of each chunk. Defaults to 800.

    Returns:
        list[str]: A list of text chunks.

    Example:
        >>> text = "This is a sample text to be split into chunks."
        >>> chunks = split_into_chunks(text, chunk_size=10)
        >>> logger.debug(chunks)
        ['This is a', 'sample', 'text to', 'be split', 'into', 'chunks.']
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        length_function=len
    )
    documents = text_splitter.create_documents([text])
    return [document.page_content for document in documents]


FILE_PATH = f"{GENERATED_DIR}/nike_2023_annual_report.txt"

with open(FILE_PATH, "r") as file:
    document_text = file.read()

chunks = split_into_chunks(document_text, chunk_size=800)

"""
## Generate descriptive document title to use in chunk header
"""
logger.info("## Generate descriptive document title to use in chunk header")

DOCUMENT_TITLE_PROMPT = """
INSTRUCTIONS
What is the title of the following document?

Your response MUST be the title of the document, and nothing else. DO NOT respond with anything else.

{document_title_guidance}

{truncation_message}

DOCUMENT
{document_text}
""".strip()

TRUNCATION_MESSAGE = """
Also note that the document text provided below is just the first ~{num_words} words of the document. That should be plenty for this task. Your response should still pertain to the entire document, not just the text provided below.
""".strip()

MAX_CONTENT_TOKENS = 4000
MODEL_NAME = "llama3.1"
TOKEN_ENCODER = tiktoken.encoding_for_model('gpt-3.5-turbo')


def make_llm_call(chat_messages: list[dict]) -> str:
    """
    Make an API call to the Ollama language model.

    Args:
        chat_messages (list[dict]): A list of message dictionaries for the chat completion.

    Returns:
        str: The generated response from the language model.
    """
#     client = Ollama(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=chat_messages,
        max_tokens=MAX_CONTENT_TOKENS,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def truncate_content(content: str, max_tokens: int) -> tuple[str, int]:
    """
    Truncate the content to a specified maximum number of tokens.

    Args:
        content (str): The input text to be truncated.
        max_tokens (int): The maximum number of tokens to keep.

    Returns:
        tuple[str, int]: A tuple containing the truncated content and the number of tokens.
    """
    tokens = TOKEN_ENCODER.encode(content, disallowed_special=())
    truncated_tokens = tokens[:max_tokens]
    return TOKEN_ENCODER.decode(truncated_tokens), min(len(tokens), max_tokens)


def get_document_title(document_text: str, document_title_guidance: str = "") -> str:
    """
    Extract the title of a document using a language model.

    Args:
        document_text (str): The text of the document.
        document_title_guidance (str, optional): Additional guidance for title extraction. Defaults to "".

    Returns:
        str: The extracted document title.
    """
    document_text, num_tokens = truncate_content(
        document_text, MAX_CONTENT_TOKENS)
    truncation_message = TRUNCATION_MESSAGE.format(
        num_words=3000) if num_tokens >= MAX_CONTENT_TOKENS else ""

    prompt = DOCUMENT_TITLE_PROMPT.format(
        document_title_guidance=document_title_guidance,
        document_text=document_text,
        truncation_message=truncation_message
    )
    chat_messages = [{"role": "user", "content": prompt}]

    return make_llm_call(chat_messages)


if __name__ == "__main__":
    document_title = get_document_title(document_text)
    logger.debug(f"Document Title: {document_title}")

"""
## Add chunk header and measure impact
Let's look at a specific example to demonstrate the impact of adding a chunk header. We'll use the Cohere reranker to measure relevance to a query with and without a chunk header.
"""
logger.info("## Add chunk header and measure impact")


def rerank_documents(query: str, chunks: List[str]) -> List[float]:
    """
    Use Cohere Rerank API to rerank the search results.

    Args:
        query (str): The search query.
        chunks (List[str]): List of document chunks to be reranked.

    Returns:
        List[float]: List of similarity scores for each chunk, in the original order.
    """
    MODEL = "rerank-english-v3.0"
    client = cohere.Client(api_key=os.environ["CO_API_KEY"])

    reranked_results = client.rerank(
        model=MODEL, query=query, documents=chunks)
    results = reranked_results.results
    reranked_indices = [result.index for result in results]
    reranked_similarity_scores = [result.relevance_score for result in results]

    similarity_scores = [0] * len(chunks)
    for i, index in enumerate(reranked_indices):
        similarity_scores[index] = reranked_similarity_scores[i]

    return similarity_scores


def compare_chunk_similarities(chunk_index: int, chunks: List[str], document_title: str, query: str) -> None:
    """
    Compare similarity scores for a chunk with and without a contextual header.

    Args:
        chunk_index (int): Index of the chunk to inspect.
        chunks (List[str]): List of all document chunks.
        document_title (str): Title of the document.
        query (str): The search query to use for comparison.

    Prints:
        Chunk header, chunk text, query, and similarity scores with and without the header.
    """
    chunk_text = chunks[chunk_index]
    chunk_wo_header = chunk_text
    chunk_w_header = f"Document Title: {document_title}\n\n{chunk_text}"

    similarity_scores = rerank_documents(
        query, [chunk_wo_header, chunk_w_header])

    logger.debug(f"\nChunk header:\nDocument Title: {document_title}")
    logger.debug(f"\nChunk text:\n{chunk_text}")
    logger.debug(f"\nQuery: {query}")
    logger.debug(
        f"\nSimilarity without contextual chunk header: {similarity_scores[0]:.4f}")
    logger.debug(
        f"Similarity with contextual chunk header: {similarity_scores[1]:.4f}")


CHUNK_INDEX_TO_INSPECT = 86
QUERY = "Nike climate change impact"

compare_chunk_similarities(CHUNK_INDEX_TO_INSPECT,
                           chunks, document_title, QUERY)

"""
This chunk is clearly about the impact of climate change on some organization, but it doesn't explicitly say "Nike" in it. So the relevance to the query "Nike climate change impact" in only about 0.1. By simply adding the document title to the chunk that similarity goes up to 0.92.

# Eval results

#### KITE

We evaluated CCH on an end-to-end RAG benchmark we created, called KITE (Knowledge-Intensive Task Evaluation).

KITE currently consists of 4 datasets and a total of 50 questions.
- **AI Papers** - ~100 academic papers about AI and RAG, downloaded from arXiv in PDF form.
- **BVP Cloud 10-Ks** - 10-Ks for all companies in the Bessemer Cloud Index (~70 of them), in PDF form.
- **Sourcegraph Company Handbook** - ~800 markdown files, with their original directory structure, downloaded from Sourcegraph's publicly accessible company handbook GitHub [page](https://github.com/sourcegraph/handbook/tree/main/content).
- **Supreme Court Opinions** - All Supreme Court opinions from Term Year 2022 (delivered from January '23 to June '23), downloaded from the official Supreme Court [website](https://www.supremecourt.gov/opinions/slipopinion/22) in PDF form.

Ground truth answers are included with each sample. Most samples also include grading rubrics. Grading is done on a scale of 0-10 for each question, with a strong LLM doing the grading.

We compare performance with and without CCH. For the CCH config we use document title and document summary. All other parameters remain the same between the two configurations. We use the Cohere 3 reranker, and we use GPT-4o for response generation.

|                         | No-CCH   | CCH          |
|-------------------------|----------|--------------|
| AI Papers               | 4.5      | 4.7          |
| BVP Cloud               | 2.6      | 6.3          |
| Sourcegraph             | 5.7      | 5.8          |
| Supreme Court Opinions  | 6.1      | 7.4          |
| **Average**             | 4.72     | 6.04         |

We can see that CCH leads to an improvement in performance on each of the four datasets. Some datasets see a large improvement while others see a small improvement. The overall average score increases from 4.72 -> 6.04, a 27.9% increase.

#### FinanceBench

We've also evaluated CCH on FinanceBench, where it contributed to a score of 83%, compared to a baseline score of 19%. For that benchmark, we tested CCH and relevant segment extraction (RSE) jointly, so we can't say exactly how much CCH contributed to that result. But the combination of CCH and RSE clearly leads to substantial accuracy improvements on FinanceBench.
"""
logger.info("# Eval results")

logger.info("\n\n[DONE]", bright=True)
