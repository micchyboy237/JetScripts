from dotenv import load_dotenv
from jet.llm.ollama.base_langchain import ChatOllama
from jet.logger import CustomLogger
from langchain.prompts import PromptTemplate
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/query_transformations.ipynb)

# Query Transformations for Improved Retrieval in RAG Systems

## Overview

This code implements three query transformation techniques to enhance the retrieval process in Retrieval-Augmented Generation (RAG) systems:

1. Query Rewriting
2. Step-back Prompting
3. Sub-query Decomposition

Each technique aims to improve the relevance and comprehensiveness of retrieved information by modifying or expanding the original query.

## Motivation

RAG systems often face challenges in retrieving the most relevant information, especially when dealing with complex or ambiguous queries. These query transformation techniques address this issue by reformulating queries to better match relevant documents or to retrieve more comprehensive information.

## Key Components

1. Query Rewriting: Reformulates queries to be more specific and detailed.
2. Step-back Prompting: Generates broader queries for better context retrieval.
3. Sub-query Decomposition: Breaks down complex queries into simpler sub-queries.

## Method Details

### 1. Query Rewriting

- **Purpose**: To make queries more specific and detailed, improving the likelihood of retrieving relevant information.
- **Implementation**:
  - Uses a GPT-4 model with a custom prompt template.
  - Takes the original query and reformulates it to be more specific and detailed.

### 2. Step-back Prompting

- **Purpose**: To generate broader, more general queries that can help retrieve relevant background information.
- **Implementation**:
  - Uses a GPT-4 model with a custom prompt template.
  - Takes the original query and generates a more general "step-back" query.

### 3. Sub-query Decomposition

- **Purpose**: To break down complex queries into simpler sub-queries for more comprehensive information retrieval.
- **Implementation**:
  - Uses a GPT-4 model with a custom prompt template.
  - Decomposes the original query into 2-4 simpler sub-queries.

## Benefits of these Approaches

1. **Improved Relevance**: Query rewriting helps in retrieving more specific and relevant information.
2. **Better Context**: Step-back prompting allows for retrieval of broader context and background information.
3. **Comprehensive Results**: Sub-query decomposition enables retrieval of information that covers different aspects of a complex query.
4. **Flexibility**: Each technique can be used independently or in combination, depending on the specific use case.

## Implementation Details

- All techniques use Ollama's GPT-4 model for query transformation.
- Custom prompt templates are used to guide the model in generating appropriate transformations.
- The code provides separate functions for each transformation technique, allowing for easy integration into existing RAG systems.

## Example Use Case

The code demonstrates each technique using the example query:
"What are the impacts of climate change on the environment?"

- **Query Rewriting** expands this to include specific aspects like temperature changes and biodiversity.
- **Step-back Prompting** generalizes it to "What are the general effects of climate change?"
- **Sub-query Decomposition** breaks it down into questions about biodiversity, oceans, weather patterns, and terrestrial environments.

## Conclusion

These query transformation techniques offer powerful ways to enhance the retrieval capabilities of RAG systems. By reformulating queries in various ways, they can significantly improve the relevance, context, and comprehensiveness of retrieved information. These methods are particularly valuable in domains where queries can be complex or multifaceted, such as scientific research, legal analysis, or comprehensive fact-finding tasks.

# Package Installation and Imports

The cell below installs all necessary packages required to run this notebook.
"""
logger.info("# Query Transformations for Improved Retrieval in RAG Systems")

# !pip install langchain langchain-openai python-dotenv



load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

"""
### 1 - Query Rewriting: Reformulating queries to improve retrieval.
"""
logger.info("### 1 - Query Rewriting: Reformulating queries to improve retrieval.")

re_write_llm = ChatOllama(model="llama3.1")

query_rewrite_template = """You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system.
Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.

Original query: {original_query}

Rewritten query:"""

query_rewrite_prompt = PromptTemplate(
    input_variables=["original_query"],
    template=query_rewrite_template
)

query_rewriter = query_rewrite_prompt | re_write_llm

def rewrite_query(original_query):
    """
    Rewrite the original query to improve retrieval.

    Args:
    original_query (str): The original user query

    Returns:
    str: The rewritten query
    """
    response = query_rewriter.invoke(original_query)
    return response.content

"""
### Demonstrate on a use case
"""
logger.info("### Demonstrate on a use case")

original_query = "What are the impacts of climate change on the environment?"
rewritten_query = rewrite_query(original_query)
logger.debug("Original query:", original_query)
logger.debug("\nRewritten query:", rewritten_query)

"""
### 2 - Step-back Prompting: Generating broader queries for better context retrieval.
"""
logger.info("### 2 - Step-back Prompting: Generating broader queries for better context retrieval.")

step_back_llm = ChatOllama(model="llama3.1")


step_back_template = """You are an AI assistant tasked with generating broader, more general queries to improve context retrieval in a RAG system.
Given the original query, generate a step-back query that is more general and can help retrieve relevant background information.

Original query: {original_query}

Step-back query:"""

step_back_prompt = PromptTemplate(
    input_variables=["original_query"],
    template=step_back_template
)

step_back_chain = step_back_prompt | step_back_llm

def generate_step_back_query(original_query):
    """
    Generate a step-back query to retrieve broader context.

    Args:
    original_query (str): The original user query

    Returns:
    str: The step-back query
    """
    response = step_back_chain.invoke(original_query)
    return response.content

"""
### Demonstrate on a use case
"""
logger.info("### Demonstrate on a use case")

original_query = "What are the impacts of climate change on the environment?"
step_back_query = generate_step_back_query(original_query)
logger.debug("Original query:", original_query)
logger.debug("\nStep-back query:", step_back_query)

"""
### 3- Sub-query Decomposition: Breaking complex queries into simpler sub-queries.
"""
logger.info("### 3- Sub-query Decomposition: Breaking complex queries into simpler sub-queries.")

sub_query_llm = ChatOllama(model="llama3.1")

subquery_decomposition_template = """You are an AI assistant tasked with breaking down complex queries into simpler sub-queries for a RAG system.
Given the original query, decompose it into 2-4 simpler sub-queries that, when answered together, would provide a comprehensive response to the original query.

Original query: {original_query}

example: What are the impacts of climate change on the environment?

Sub-queries:
1. What are the impacts of climate change on biodiversity?
2. How does climate change affect the oceans?
3. What are the effects of climate change on agriculture?
4. What are the impacts of climate change on human health?"""


subquery_decomposition_prompt = PromptTemplate(
    input_variables=["original_query"],
    template=subquery_decomposition_template
)

subquery_decomposer_chain = subquery_decomposition_prompt | sub_query_llm

def decompose_query(original_query: str):
    """
    Decompose the original query into simpler sub-queries.

    Args:
    original_query (str): The original complex query

    Returns:
    List[str]: A list of simpler sub-queries
    """
    response = subquery_decomposer_chain.invoke(original_query).content
    sub_queries = [q.strip() for q in response.split('\n') if q.strip() and not q.strip().startswith('Sub-queries:')]
    return sub_queries

"""
### Demonstrate on a use case
"""
logger.info("### Demonstrate on a use case")

original_query = "What are the impacts of climate change on the environment?"
sub_queries = decompose_query(original_query)
logger.debug("\nSub-queries:")
for i, sub_query in enumerate(sub_queries, 1):
    logger.debug(sub_query)

logger.info("\n\n[DONE]", bright=True)