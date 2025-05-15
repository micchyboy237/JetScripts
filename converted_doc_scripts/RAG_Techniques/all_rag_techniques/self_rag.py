from dotenv import load_dotenv
from evaluation.evalute_rag import *
from helper_functions import *
from jet.llm.ollama.base_langchain import ChatOllama
from jet.logger import CustomLogger
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
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
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/self_rag.ipynb)

# Self-RAG: A Dynamic Approach to Retrieval-Augmented Generation

## Overview

Self-RAG is an advanced algorithm that combines the power of retrieval-based and generation-based approaches in natural language processing. It dynamically decides whether to use retrieved information and how to best utilize it in generating responses, aiming to produce more accurate, relevant, and useful outputs.

## Motivation

Traditional question-answering systems often struggle with balancing the use of retrieved information and the generation of new content. Some systems might rely too heavily on retrieved data, leading to responses that lack flexibility, while others might generate responses without sufficient grounding in factual information. Self-RAG addresses these issues by implementing a multi-step process that carefully evaluates the necessity and relevance of retrieved information, and assesses the quality of generated responses.

## Key Components

1. **Retrieval Decision**: Determines if retrieval is necessary for a given query.
2. **Document Retrieval**: Fetches potentially relevant documents from a vector store.
3. **Relevance Evaluation**: Assesses the relevance of retrieved documents to the query.
4. **Response Generation**: Generates responses based on relevant contexts.
5. **Support Assessment**: Evaluates how well the generated response is supported by the context.
6. **Utility Evaluation**: Rates the usefulness of the generated response.

## Method Details

1. **Retrieval Decision**: The algorithm first decides if retrieval is necessary for the given query. This step prevents unnecessary retrieval for queries that can be answered directly.

2. **Document Retrieval**: If retrieval is deemed necessary, the algorithm fetches the top-k most similar documents from a vector store.

3. **Relevance Evaluation**: Each retrieved document is evaluated for its relevance to the query. This step filters out irrelevant information, ensuring that only pertinent context is used for generation.

4. **Response Generation**: The algorithm generates responses using the relevant contexts. If no relevant contexts are found, it generates a response without retrieval.

5. **Support Assessment**: Each generated response is evaluated to determine how well it is supported by the context. This step helps in identifying responses that are grounded in the provided information.

6. **Utility Evaluation**: The utility of each response is rated, considering how well it addresses the original query.

7. **Response Selection**: The final step involves selecting the best response based on the support assessment and utility evaluation.

## Benefits of the Approach

1. **Dynamic Retrieval**: By deciding whether retrieval is necessary, the system can adapt to different types of queries efficiently.

2. **Relevance Filtering**: The relevance evaluation step ensures that only pertinent information is used, reducing noise in the generation process.

3. **Quality Assurance**: The support assessment and utility evaluation provide a way to gauge the quality of generated responses.

4. **Flexibility**: The system can generate responses with or without retrieval, adapting to the available information.

5. **Improved Accuracy**: By grounding responses in relevant retrieved information and assessing their support, the system can produce more accurate outputs.

## Conclusion

Self-RAG represents a sophisticated approach to question-answering and information retrieval tasks. By incorporating multiple evaluation steps and dynamically deciding on the use of retrieved information, it aims to produce responses that are not only relevant and accurate but also useful to the end-user. This method showcases the potential of combining retrieval and generation techniques in a thoughtful, evaluated manner to enhance the quality of AI-generated responses.

<div style="text-align: center;">

<img src="../images/self_rag.svg" alt="Self RAG" style="width:80%; height:auto;">
</div>

# Package Installation and Imports

The cell below installs all necessary packages required to run this notebook.
"""
logger.info("# Self-RAG: A Dynamic Approach to Retrieval-Augmented Generation")

# !pip install langchain langchain-openai python-dotenv

# !git clone https://github.com/N7/RAG_TECHNIQUES.git
sys.path.append('RAG_TECHNIQUES')


load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

"""
### Define files path
"""
logger.info("### Define files path")

os.makedirs('data', exist_ok=True)

# !wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/N7/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
# !wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/N7/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf

path = f"{GENERATED_DIR}/Understanding_Climate_Change.pdf"

"""
### Create a vector store
"""
logger.info("### Create a vector store")

vectorstore = encode_pdf(path)

"""
### Initialize the language model
"""
logger.info("### Initialize the language model")

llm = ChatOllama(model="llama3.1")

"""
### Defining prompt templates
"""
logger.info("### Defining prompt templates")


class RetrievalResponse(BaseModel):
    response: str = Field(..., title="Determines if retrieval is necessary",
                          description="Output only 'Yes' or 'No'.")


retrieval_prompt = PromptTemplate(
    input_variables=["query"],
    template="Given the query '{query}', determine if retrieval is necessary. Output only 'Yes' or 'No'."
)


class RelevanceResponse(BaseModel):
    response: str = Field(..., title="Determines if context is relevant",
                          description="Output only 'Relevant' or 'Irrelevant'.")


relevance_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="Given the query '{query}' and the context '{context}', determine if the context is relevant. Output only 'Relevant' or 'Irrelevant'."
)


class GenerationResponse(BaseModel):
    response: str = Field(..., title="Generated response",
                          description="The generated response.")


generation_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="Given the query '{query}' and the context '{context}', generate a response."
)


class SupportResponse(BaseModel):
    response: str = Field(..., title="Determines if response is supported",
                          description="Output 'Fully supported', 'Partially supported', or 'No support'.")


support_prompt = PromptTemplate(
    input_variables=["response", "context"],
    template="Given the response '{response}' and the context '{context}', determine if the response is supported by the context. Output 'Fully supported', 'Partially supported', or 'No support'."
)


class UtilityResponse(BaseModel):
    response: int = Field(..., title="Utility rating",
                          description="Rate the utility of the response from 1 to 5.")


utility_prompt = PromptTemplate(
    input_variables=["query", "response"],
    template="Given the query '{query}' and the response '{response}', rate the utility of the response from 1 to 5."
)

retrieval_chain = retrieval_prompt | llm.with_structured_output(
    RetrievalResponse)
relevance_chain = relevance_prompt | llm.with_structured_output(
    RelevanceResponse)
generation_chain = generation_prompt | llm.with_structured_output(
    GenerationResponse)
support_chain = support_prompt | llm.with_structured_output(SupportResponse)
utility_chain = utility_prompt | llm.with_structured_output(UtilityResponse)

"""
### Defining the self RAG logic flow
"""
logger.info("### Defining the self RAG logic flow")


def self_rag(query, vectorstore, top_k=3):
    logger.debug(f"\nProcessing query: {query}")

    logger.debug("Step 1: Determining if retrieval is necessary...")
    input_data = {"query": query}
    retrieval_decision = retrieval_chain.invoke(
        input_data).response.strip().lower()
    logger.debug(f"Retrieval decision: {retrieval_decision}")

    if retrieval_decision == 'yes':
        logger.debug("Step 2: Retrieving relevant documents...")
        docs = vectorstore.similarity_search(query, k=top_k)
        contexts = [doc.page_content for doc in docs]
        logger.debug(f"Retrieved {len(contexts)} documents")

        logger.debug("Step 3: Evaluating relevance of retrieved documents...")
        relevant_contexts = []
        for i, context in enumerate(contexts):
            input_data = {"query": query, "context": context}
            relevance = relevance_chain.invoke(
                input_data).response.strip().lower()
            logger.debug(f"Document {i+1} relevance: {relevance}")
            if relevance == 'relevant':
                relevant_contexts.append(context)

        logger.debug(f"Number of relevant contexts: {len(relevant_contexts)}")

        if not relevant_contexts:
            logger.debug(
                "No relevant contexts found. Generating without retrieval...")
            input_data = {"query": query,
                          "context": "No relevant context found."}
            return generation_chain.invoke(input_data).response

        logger.debug("Step 4: Generating responses using relevant contexts...")
        responses = []
        for i, context in enumerate(relevant_contexts):
            logger.debug(f"Generating response for context {i+1}...")
            input_data = {"query": query, "context": context}
            response = generation_chain.invoke(input_data).response

            logger.debug(f"Step 5: Assessing support for response {i+1}...")
            input_data = {"response": response, "context": context}
            support = support_chain.invoke(input_data).response.strip().lower()
            logger.debug(f"Support assessment: {support}")

            logger.debug(f"Step 6: Evaluating utility for response {i+1}...")
            input_data = {"query": query, "response": response}
            utility = int(utility_chain.invoke(input_data).response)
            logger.debug(f"Utility score: {utility}")

            responses.append((response, support, utility))

        logger.debug("Selecting the best response...")
        best_response = max(responses, key=lambda x: (
            x[1] == 'fully supported', x[2]))
        logger.debug(
            f"Best response support: {best_response[1]}, utility: {best_response[2]}")
        return best_response[0]
    else:
        logger.debug("Generating without retrieval...")
        input_data = {"query": query, "context": "No retrieval necessary."}
        return generation_chain.invoke(input_data).response


"""
### Test the self-RAG function easy query with high relevance
"""
logger.info("### Test the self-RAG function easy query with high relevance")

query = "What is the impact of climate change on the environment?"
response = self_rag(query, vectorstore)

logger.debug("\nFinal response:")
logger.debug(response)

"""
### Test the self-RAG function with a more challenging query with low relevance
"""
logger.info(
    "### Test the self-RAG function with a more challenging query with low relevance")

query = "how did harry beat quirrell?"
response = self_rag(query, vectorstore)

logger.debug("\nFinal response:")
logger.debug(response)

logger.info("\n\n[DONE]", bright=True)
