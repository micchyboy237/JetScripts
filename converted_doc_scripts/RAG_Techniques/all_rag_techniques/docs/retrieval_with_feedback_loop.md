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

```python
# Install required packages
!pip install langchain langchain-openai python-dotenv
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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import json
from typing import List, Dict, Any


# Original path append replaced for Colab compatibility
from helper_functions import *
from evaluation.evalute_rag import *

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
```

### Define documents path

```python
# Download required data files
import os
os.makedirs('data', exist_ok=True)

# Download the PDF document used in this notebook
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/N7/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
!wget -O data/feedback_data.json https://raw.githubusercontent.com/N7/RAG_TECHNIQUES/main/data/feedback_data.json
!wget -O data/Understanding_Climate_Change.pdf https://raw.githubusercontent.com/N7/RAG_TECHNIQUES/main/data/Understanding_Climate_Change.pdf
```

```python
path = "data/Understanding_Climate_Change.pdf"
```

### Create vector store and retrieval QA chain

```python
content = read_pdf_to_string(path)
vectorstore = encode_from_string(content)
retriever = vectorstore.as_retriever()

llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
```

### Function to format user feedback in a dictionary

```python
def get_user_feedback(query, response, relevance, quality, comments=""):
    return {
        "query": query,
        "response": response,
        "relevance": int(relevance),
        "quality": int(quality),
        "comments": comments
    }
```

### Function to store the feedback in a json file

```python
def store_feedback(feedback):
    with open("data/feedback_data.json", "a") as f:
        json.dump(feedback, f)
        f.write("\n")
```

### Function to read the feedback file

```python
def load_feedback_data():
    feedback_data = []
    try:
        with open("data/feedback_data.json", "r") as f:
            for line in f:
                feedback_data.append(json.loads(line.strip()))
    except FileNotFoundError:
        print("No feedback data file found. Starting with empty feedback.")
    return feedback_data
```

### Function to adjust files relevancy based on the feedbacks file

```python
class Response(BaseModel):
    answer: str = Field(..., title="The answer to the question. The options can be only 'Yes' or 'No'")

def adjust_relevance_scores(query: str, docs: List[Any], feedback_data: List[Dict[str, Any]]) -> List[Any]:
    # Create a prompt template for relevance checking
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
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)

    # Create an LLMChain for relevance checking
    relevance_chain = relevance_prompt | llm.with_structured_output(Response)

    for doc in docs:
        relevant_feedback = []
        
        for feedback in feedback_data:
            # Use LLM to check relevance
            input_data = {
                "query": query,
                "feedback_query": feedback['query'],
                "doc_content": doc.page_content[:1000],
                "feedback_response": feedback['response']
            }
            result = relevance_chain.invoke(input_data).answer
            
            if result == 'yes':
                relevant_feedback.append(feedback)
        
        # Adjust the relevance score based on feedback
        if relevant_feedback:
            avg_relevance = sum(f['relevance'] for f in relevant_feedback) / len(relevant_feedback)
            doc.metadata['relevance_score'] *= (avg_relevance / 3)  # Assuming a 1-5 scale, 3 is neutral
    
    # Re-rank documents based on adjusted scores
    return sorted(docs, key=lambda x: x.metadata['relevance_score'], reverse=True)
```

### Function to fine tune the vector index to include also queries + answers that received good feedbacks

```python
def fine_tune_index(feedback_data: List[Dict[str, Any]], texts: List[str]) -> Any:
    # Filter high-quality responses
    good_responses = [f for f in feedback_data if f['relevance'] >= 4 and f['quality'] >= 4]
    
    # Extract queries and responses, and create new documents
    additional_texts = []
    for f in good_responses:
        combined_text = f['query'] + " " + f['response']
        additional_texts.append(combined_text)

    # make the list a string
    additional_texts = " ".join(additional_texts)
    
    # Create a new index with original and high-quality texts
    all_texts = texts + additional_texts
    new_vectorstore = encode_from_string(all_texts)
    
    return new_vectorstore
```

### Demonstration of how to retrieve answers with respect to user feedbacks

```python

query = "What is the greenhouse effect?"

# Get response from RAG system
response = qa_chain(query)["result"]

relevance = 5
quality = 5

# Collect feedback
feedback = get_user_feedback(query, response, relevance, quality)

# Store feedback
store_feedback(feedback)

# Adjust relevance scores for future retrievals
docs = retriever.get_relevant_documents(query)
adjusted_docs = adjust_relevance_scores(query, docs, load_feedback_data())

# Update the retriever with adjusted docs
retriever.search_kwargs['k'] = len(adjusted_docs)
retriever.search_kwargs['docs'] = adjusted_docs
```

### Finetune the vectorstore periodicly

```python
# Periodically (e.g., daily or weekly), fine-tune the index
new_vectorstore = fine_tune_index(load_feedback_data(), content)
retriever = new_vectorstore.as_retriever()
```
