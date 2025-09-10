from dotenv import load_dotenv
from evaluation.evalute_rag import *
from helper_functions import *
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import CustomLogger
from langchain.prompts import PromptTemplate
from langchain.tools import DuckDuckGoSearchResults
from langchain_core.pydantic_v1 import BaseModel, Field
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

"""
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/crag.ipynb)

# Corrective RAG Process: Retrieval-Augmented Generation with Dynamic Correction

## Overview

The Corrective RAG (Retrieval-Augmented Generation) process is an advanced information retrieval and response generation system. It extends the standard RAG approach by dynamically evaluating and correcting the retrieval process, combining the power of vector databases, web search, and language models to provide accurate and context-aware responses to user queries.

## Motivation

While traditional RAG systems have improved information retrieval and response generation, they can still fall short when the retrieved information is irrelevant or outdated. The Corrective RAG process addresses these limitations by:

1. Leveraging pre-existing knowledge bases
2. Evaluating the relevance of retrieved information
3. Dynamically searching the web when necessary
4. Refining and combining knowledge from multiple sources
5. Generating human-like responses based on the most appropriate knowledge

## Key Components

1. **FAISS Index**: A vector database for efficient similarity search of pre-existing knowledge.
2. **Retrieval Evaluator**: Assesses the relevance of retrieved documents to the query.
3. **Knowledge Refinement**: Extracts key information from documents when necessary.
4. **Web Search Query Rewriter**: Optimizes queries for web searches when local knowledge is insufficient.
5. **Response Generator**: Creates human-like responses based on the accumulated knowledge.

## Method Details

1. **Document Retrieval**: 
   - Performs similarity search in the FAISS index to find relevant documents.
   - Retrieves top-k documents (default k=3).

2. **Document Evaluation**:
   - Calculates relevance scores for each retrieved document.
   - Determines the best course of action based on the highest relevance score.

3. **Corrective Knowledge Acquisition**:
   - If high relevance (score > 0.7): Uses the most relevant document as-is.
   - If low relevance (score < 0.3): Corrects by performing a web search with a rewritten query.
   - If ambiguous (0.3 ≤ score ≤ 0.7): Corrects by combining the most relevant document with web search results.

4. **Adaptive Knowledge Processing**:
   - For web search results: Refines the knowledge to extract key points.
   - For ambiguous cases: Combines raw document content with refined web search results.

5. **Response Generation**:
   - Uses a language model to generate a human-like response based on the query and acquired knowledge.
   - Includes source information in the response for transparency.

## Benefits of the Corrective RAG Approach

1. **Dynamic Correction**: Adapts to the quality of retrieved information, ensuring relevance and accuracy.
2. **Flexibility**: Leverages both pre-existing knowledge and web search as needed.
3. **Accuracy**: Evaluates the relevance of information before using it, ensuring high-quality responses.
4. **Transparency**: Provides source information, allowing users to verify the origin of the information.
5. **Efficiency**: Uses vector search for quick retrieval from large knowledge bases.
6. **Contextual Understanding**: Combines multiple sources of information when necessary to provide comprehensive responses.
7. **Up-to-date Information**: Can supplement or replace outdated local knowledge with current web information.

## Conclusion

The Corrective RAG process represents a sophisticated evolution of the standard RAG approach. By intelligently evaluating and correcting the retrieval process, it overcomes common limitations of traditional RAG systems. This dynamic approach ensures that responses are based on the most relevant and up-to-date information available, whether from local knowledge bases or the web. The system's ability to adapt its information sourcing strategy based on relevance scores makes it particularly suited for applications requiring high accuracy and current information, such as research assistance, dynamic knowledge bases, and advanced question-answering systems.

<div style="text-align: center;">

<img src="../images/crag.svg" alt="Corrective RAG" style="width:80%; height:auto;">
</div>

# Package Installation and Imports

The cell below installs all necessary packages required to run this notebook.
"""
logger.info(
    "# Corrective RAG Process: Retrieval-Augmented Generation with Dynamic Correction")

# !pip install langchain langchain-openai python-dotenv

# !git clone https://github.com/NirDiamant/RAG_TECHNIQUES.git
sys.path.append('RAG_TECHNIQUES')


load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

"""
### Define files path
"""
logger.info("### Define files path")

os.makedirs('data', exist_ok=True)

# !wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
# !wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/NirDiamant/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf

path = f"{GENERATED_DIR}/Understanding_Climate_Change.pdf"

"""
### Create a vector store
"""
logger.info("### Create a vector store")

vectorstore = encode_pdf(path)

"""
### Initialize Ollama language model
"""
logger.info("### Initialize Ollama language model")

llm = ChatOllama(model="llama3.1")

"""
### Initialize search tool
"""
logger.info("### Initialize search tool")

search = DuckDuckGoSearchResults()

"""
### Define retrieval evaluator, knowledge refinement and query rewriter llm chains
"""
logger.info(
    "### Define retrieval evaluator, knowledge refinement and query rewriter llm chains")


class RetrievalEvaluatorInput(BaseModel):
    relevance_score: float = Field(
        ..., description="The relevance score of the document to the query. the score should be between 0 and 1.")


def retrieval_evaluator(query: str, document: str) -> float:
    prompt = PromptTemplate(
        input_variables=["query", "document"],
        template="On a scale from 0 to 1, how relevant is the following document to the query? Query: {query}\nDocument: {document}\nRelevance score:"
    )
    chain = prompt | llm.with_structured_output(RetrievalEvaluatorInput)
    input_variables = {"query": query, "document": document}
    result = chain.invoke(input_variables).relevance_score
    return result


class KnowledgeRefinementInput(BaseModel):
    key_points: str = Field(...,
                            description="The document to extract key information from.")


def knowledge_refinement(document: str) -> List[str]:
    prompt = PromptTemplate(
        input_variables=["document"],
        template="Extract the key information from the following document in bullet points:\n{document}\nKey points:"
    )
    chain = prompt | llm.with_structured_output(KnowledgeRefinementInput)
    input_variables = {"document": document}
    result = chain.invoke(input_variables).key_points
    return [point.strip() for point in result.split('\n') if point.strip()]


class QueryRewriterInput(BaseModel):
    query: str = Field(..., description="The query to rewrite.")


def rewrite_query(query: str) -> str:
    prompt = PromptTemplate(
        input_variables=["query"],
        template="Rewrite the following query to make it more suitable for a web search:\n{query}\nRewritten query:"
    )
    chain = prompt | llm.with_structured_output(QueryRewriterInput)
    input_variables = {"query": query}
    return chain.invoke(input_variables).query.strip()


"""
### Helper function to parse search results
"""
logger.info("### Helper function to parse search results")


def parse_search_results(results_string: str) -> List[Tuple[str, str]]:
    """
    Parse a JSON string of search results into a list of title-link tuples.

    Args:
        results_string (str): A JSON-formatted string containing search results.

    Returns:
        List[Tuple[str, str]]: A list of tuples, where each tuple contains the title and link of a search result.
                               If parsing fails, an empty list is returned.
    """
    try:
        results = json.loads(results_string)
        return [(result.get('title', 'Untitled'), result.get('link', '')) for result in results]
    except json.JSONDecodeError:
        logger.debug("Error parsing search results. Returning empty list.")
        return []


"""
### Define sub functions for the CRAG process
"""
logger.info("### Define sub functions for the CRAG process")


def retrieve_documents(query: str, faiss_index: FAISS, k: int = 3) -> List[str]:
    """
    Retrieve documents based on a query using a FAISS index.

    Args:
        query (str): The query string to search for.
        faiss_index (FAISS): The FAISS index used for similarity search.
        k (int): The number of top documents to retrieve. Defaults to 3.

    Returns:
        List[str]: A list of the retrieved document contents.
    """
    docs = faiss_index.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]


def evaluate_documents(query: str, documents: List[str]) -> List[float]:
    """
    Evaluate the relevance of documents based on a query.

    Args:
        query (str): The query string.
        documents (List[str]): A list of document contents to evaluate.

    Returns:
        List[float]: A list of relevance scores for each document.
    """
    return [retrieval_evaluator(query, doc) for doc in documents]


def perform_web_search(query: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Perform a web search based on a query.

    Args:
        query (str): The query string to search for.

    Returns:
        Tuple[List[str], List[Tuple[str, str]]]:
            - A list of refined knowledge obtained from the web search.
            - A list of tuples containing titles and links of the sources.
    """
    rewritten_query = rewrite_query(query)
    web_results = search.run(rewritten_query)
    web_knowledge = knowledge_refinement(web_results)
    sources = parse_search_results(web_results)
    return web_knowledge, sources


def generate_response(query: str, knowledge: str, sources: List[Tuple[str, str]]) -> str:
    """
    Generate a response to a query using knowledge and sources.

    Args:
        query (str): The query string.
        knowledge (str): The refined knowledge to use in the response.
        sources (List[Tuple[str, str]]): A list of tuples containing titles and links of the sources.

    Returns:
        str: The generated response.
    """
    response_prompt = PromptTemplate(
        input_variables=["query", "knowledge", "sources"],
        template="Based on the following knowledge, answer the query. Include the sources with their links (if available) at the end of your answer:\nQuery: {query}\nKnowledge: {knowledge}\nSources: {sources}\nAnswer:"
    )
    input_variables = {
        "query": query,
        "knowledge": knowledge,
        "sources": "\n".join([f"{title}: {link}" if link else title for title, link in sources])
    }
    response_chain = response_prompt | llm
    return response_chain.invoke(input_variables).content


"""
### CRAG process
"""
logger.info("### CRAG process")


def crag_process(query: str, faiss_index: FAISS) -> str:
    """
    Process a query by retrieving, evaluating, and using documents or performing a web search to generate a response.

    Args:
        query (str): The query string to process.
        faiss_index (FAISS): The FAISS index used for document retrieval.

    Returns:
        str: The generated response based on the query.
    """
    logger.debug(f"\nProcessing query: {query}")

    retrieved_docs = retrieve_documents(query, faiss_index)
    eval_scores = evaluate_documents(query, retrieved_docs)

    logger.debug(f"\nRetrieved {len(retrieved_docs)} documents")
    logger.debug(f"Evaluation scores: {eval_scores}")

    max_score = max(eval_scores)
    sources = []

    if max_score > 0.7:
        logger.debug("\nAction: Correct - Using retrieved document")
        best_doc = retrieved_docs[eval_scores.index(max_score)]
        final_knowledge = best_doc
        sources.append(("Retrieved document", ""))
    elif max_score < 0.3:
        logger.debug("\nAction: Incorrect - Performing web search")
        final_knowledge, sources = perform_web_search(query)
    else:
        logger.debug(
            "\nAction: Ambiguous - Combining retrieved document and web search")
        best_doc = retrieved_docs[eval_scores.index(max_score)]
        retrieved_knowledge = knowledge_refinement(best_doc)
        web_knowledge, web_sources = perform_web_search(query)
        final_knowledge = "\n".join(retrieved_knowledge + web_knowledge)
        sources = [("Retrieved document", "")] + web_sources

    logger.debug("\nFinal knowledge:")
    logger.debug(final_knowledge)

    logger.debug("\nSources:")
    for title, link in sources:
        logger.debug(f"{title}: {link}" if link else title)

    logger.debug("\nGenerating response...")
    response = generate_response(query, final_knowledge, sources)

    logger.debug("\nResponse generated")
    return response


"""
### Example query with high relevance to the document
"""
logger.info("### Example query with high relevance to the document")

query = "What are the main causes of climate change?"
result = crag_process(query, vectorstore)
logger.debug(f"Query: {query}")
logger.debug(f"Answer: {result}")

"""
### Example query with low relevance to the document
"""
logger.info("### Example query with low relevance to the document")

query = "how did harry beat quirrell?"
result = crag_process(query, vectorstore)
logger.debug(f"Query: {query}")
logger.debug(f"Answer: {result}")

logger.info("\n\n[DONE]", bright=True)
