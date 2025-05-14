from dotenv import load_dotenv
from evaluation.evalute_rag import *
from helper_functions import *
from jet.llm.ollama.base_langchain import ChatOllama
from jet.logger import CustomLogger
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import json
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
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/retrieval_with_feedback_loop.ipynb)

# RAG System with Feedback Loop: Enhancing Retrieval and Response Quality

## Overview

This system implements a Retrieval-Augmented Generation (RAG) approach with an integrated feedback loop. It aims to improve the quality and relevance of responses over time by incorporating user feedback and dynamically adjusting the retrieval process.

## Motivation

Traditional RAG systems can sometimes produce inconsistent or irrelevant responses due to limitations in the retrieval process or the underlying knowledge base. By implementing a feedback loop, we can:

1. Continuously improve the quality of retrieved documents
2. Enhance the relevance of generated responses
3. Adapt the system to user preferences and needs over time

## Key Components

1. **PDF Content Extraction**: Extracts text from PDF documents
2. **Vectorstore**: Stores and indexes document embeddings for efficient retrieval
3. **Retriever**: Fetches relevant documents based on user queries
4. **Language Model**: Generates responses using retrieved documents
5. **Feedback Collection**: Gathers user feedback on response quality and relevance
6. **Feedback Storage**: Persists user feedback for future use
7. **Relevance Score Adjustment**: Modifies document relevance based on feedback
8. **Index Fine-tuning**: Periodically updates the vectorstore using accumulated feedback

## Method Details

### 1. Initial Setup
- The system reads PDF content and creates a vectorstore
- A retriever is initialized using the vectorstore
- A language model (LLM) is set up for response generation

### 2. Query Processing
- When a user submits a query, the retriever fetches relevant documents
- The LLM generates a response based on the retrieved documents

### 3. Feedback Collection
- The system collects user feedback on the response's relevance and quality
- Feedback is stored in a JSON file for persistence

### 4. Relevance Score Adjustment
- For subsequent queries, the system loads previous feedback
- An LLM evaluates the relevance of past feedback to the current query
- Document relevance scores are adjusted based on this evaluation

### 5. Retriever Update
- The retriever is updated with the adjusted document scores
- This ensures that future retrievals benefit from past feedback

### 6. Periodic Index Fine-tuning
- At regular intervals, the system fine-tunes the index
- High-quality feedback is used to create additional documents
- The vectorstore is updated with these new documents, improving overall retrieval quality

## Benefits of this Approach

1. **Continuous Improvement**: The system learns from each interaction, gradually enhancing its performance.
2. **Personalization**: By incorporating user feedback, the system can adapt to individual or group preferences over time.
3. **Increased Relevance**: The feedback loop helps prioritize more relevant documents in future retrievals.
4. **Quality Control**: Low-quality or irrelevant responses are less likely to be repeated as the system evolves.
5. **Adaptability**: The system can adjust to changes in user needs or document contents over time.

## Conclusion

This RAG system with a feedback loop represents a significant advancement over traditional RAG implementations. By continuously learning from user interactions, it offers a more dynamic, adaptive, and user-centric approach to information retrieval and response generation. This system is particularly valuable in domains where information accuracy and relevance are critical, and where user needs may evolve over time.

While the implementation adds complexity compared to a basic RAG system, the benefits in terms of response quality and user satisfaction make it a worthwhile investment for applications requiring high-quality, context-aware information retrieval and generation.

<div style="text-align: center;">

<img src="../images/retrieval_with_feedback_loop.svg" alt="retrieval with feedback loop" style="width:40%; height:auto;">
</div>

# Package Installation and Imports

The cell below installs all necessary packages required to run this notebook.
"""
logger.info("# RAG System with Feedback Loop: Enhancing Retrieval and Response Quality")

# !pip install langchain langchain-openai python-dotenv

# !git clone https://github.com/N7/RAG_TECHNIQUES.git
sys.path.append('RAG_TECHNIQUES')




load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

"""
### Define documents path
"""
logger.info("### Define documents path")

os.makedirs('data', exist_ok=True)

# !wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/N7/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
# !wget -O data/feedback_data.json https://raw.githubusercontent.com/N7/RAG_TECHNIQUES/main/data/feedback_data.json
# !wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/N7/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf

path = f"{GENERATED_DIR}/Understanding_Climate_Change.pdf"

"""
### Create vector store and retrieval QA chain
"""
logger.info("### Create vector store and retrieval QA chain")

content = read_pdf_to_string(path)
vectorstore = encode_from_string(content)
retriever = vectorstore.as_retriever()

llm = ChatOllama(model="llama3.1")
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

"""
### Function to format user feedback in a dictionary
"""
logger.info("### Function to format user feedback in a dictionary")

def get_user_feedback(query, response, relevance, quality, comments=""):
    return {
        "query": query,
        "response": response,
        "relevance": int(relevance),
        "quality": int(quality),
        "comments": comments
    }

"""
### Function to store the feedback in a json file
"""
logger.info("### Function to store the feedback in a json file")

def store_feedback(feedback):
    with open(f"{GENERATED_DIR}/feedback_data.json", "a") as f:
        json.dump(feedback, f)
        f.write("\n")

"""
### Function to read the feedback file
"""
logger.info("### Function to read the feedback file")

def load_feedback_data():
    feedback_data = []
    try:
        with open(f"{GENERATED_DIR}/feedback_data.json", "r") as f:
            for line in f:
                feedback_data.append(json.loads(line.strip()))
    except FileNotFoundError:
        logger.debug("No feedback data file found. Starting with empty feedback.")
    return feedback_data

"""
### Function to adjust files relevancy based on the feedbacks file
"""
logger.info("### Function to adjust files relevancy based on the feedbacks file")

class Response(BaseModel):
    answer: str = Field(..., title="The answer to the question. The options can be only 'Yes' or 'No'")

def adjust_relevance_scores(query: str, docs: List[Any], feedback_data: List[Dict[str, Any]]) -> List[Any]:
    relevance_prompt = PromptTemplate(
        input_variables=["query", "feedback_query", "doc_content", "feedback_response"],
        template="""
        Determine if the following feedback response is relevant to the current query and document content.
        You are also provided with the Feedback original query that was used to generate the feedback response.
        Current query: {query}
        Feedback query: {feedback_query}
        Document content: {doc_content}
        Feedback response: {feedback_response}

        Is this feedback relevant? Respond with only 'Yes' or 'No'.
        """
    )
    llm = ChatOllama(model="llama3.1")

    relevance_chain = relevance_prompt | llm.with_structured_output(Response)

    for doc in docs:
        relevant_feedback = []

        for feedback in feedback_data:
            input_data = {
                "query": query,
                "feedback_query": feedback['query'],
                "doc_content": doc.page_content[:1000],
                "feedback_response": feedback['response']
            }
            result = relevance_chain.invoke(input_data).answer

            if result == 'yes':
                relevant_feedback.append(feedback)

        if relevant_feedback:
            avg_relevance = sum(f['relevance'] for f in relevant_feedback) / len(relevant_feedback)
            doc.metadata['relevance_score'] *= (avg_relevance / 3)  # Assuming a 1-5 scale, 3 is neutral

    return sorted(docs, key=lambda x: x.metadata['relevance_score'], reverse=True)

"""
### Function to fine tune the vector index to include also queries + answers that received good feedbacks
"""
logger.info("### Function to fine tune the vector index to include also queries + answers that received good feedbacks")

def fine_tune_index(feedback_data: List[Dict[str, Any]], texts: List[str]) -> Any:
    good_responses = [f for f in feedback_data if f['relevance'] >= 4 and f['quality'] >= 4]

    additional_texts = []
    for f in good_responses:
        combined_text = f['query'] + " " + f['response']
        additional_texts.append(combined_text)

    additional_texts = " ".join(additional_texts)

    all_texts = texts + additional_texts
    new_vectorstore = encode_from_string(all_texts)

    return new_vectorstore

"""
### Demonstration of how to retrieve answers with respect to user feedbacks
"""
logger.info("### Demonstration of how to retrieve answers with respect to user feedbacks")

query = "What is the greenhouse effect?"

response = qa_chain(query)["result"]

relevance = 5
quality = 5

feedback = get_user_feedback(query, response, relevance, quality)

store_feedback(feedback)

docs = retriever.get_relevant_documents(query)
adjusted_docs = adjust_relevance_scores(query, docs, load_feedback_data())

retriever.search_kwargs['k'] = len(adjusted_docs)
retriever.search_kwargs['docs'] = adjusted_docs

"""
### Finetune the vectorstore periodicly
"""
logger.info("### Finetune the vectorstore periodicly")

new_vectorstore = fine_tune_index(load_feedback_data(), content)
retriever = new_vectorstore.as_retriever()

logger.info("\n\n[DONE]", bright=True)