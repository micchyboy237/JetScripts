from dotenv import load_dotenv
from evaluation.evalute_rag import *
from helper_functions import *
from jet.llm.ollama.base_langchain import ChatOllama
from jet.logger import CustomLogger
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any, Tuple
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
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/reranking.ipynb)

# Reranking Methods in RAG Systems

## Overview
Reranking is a crucial step in Retrieval-Augmented Generation (RAG) systems that aims to improve the relevance and quality of retrieved documents. It involves reassessing and reordering initially retrieved documents to ensure that the most pertinent information is prioritized for subsequent processing or presentation.

## Motivation
The primary motivation for reranking in RAG systems is to overcome limitations of initial retrieval methods, which often rely on simpler similarity metrics. Reranking allows for more sophisticated relevance assessment, taking into account nuanced relationships between queries and documents that might be missed by traditional retrieval techniques. This process aims to enhance the overall performance of RAG systems by ensuring that the most relevant information is used in the generation phase.

## Key Components
Reranking systems typically include the following components:

1. Initial Retriever: Often a vector store using embedding-based similarity search.
2. Reranking Model: This can be either:
   - A Large Language Model (LLM) for scoring relevance
   - A Cross-Encoder model specifically trained for relevance assessment
3. Scoring Mechanism: A method to assign relevance scores to documents
4. Sorting and Selection Logic: To reorder documents based on new scores

## Method Details
The reranking process generally follows these steps:

1. Initial Retrieval: Fetch an initial set of potentially relevant documents.
2. Pair Creation: Form query-document pairs for each retrieved document.
3. Scoring: 
   - LLM Method: Use prompts to ask the LLM to rate document relevance.
   - Cross-Encoder Method: Feed query-document pairs directly into the model.
4. Score Interpretation: Parse and normalize the relevance scores.
5. Reordering: Sort documents based on their new relevance scores.
6. Selection: Choose the top K documents from the reordered list.

## Benefits of this Approach
Reranking offers several advantages:

1. Improved Relevance: By using more sophisticated models, reranking can capture subtle relevance factors.
2. Flexibility: Different reranking methods can be applied based on specific needs and resources.
3. Enhanced Context Quality: Providing more relevant documents to the RAG system improves the quality of generated responses.
4. Reduced Noise: Reranking helps filter out less relevant information, focusing on the most pertinent content.

## Conclusion
Reranking is a powerful technique in RAG systems that significantly enhances the quality of retrieved information. Whether using LLM-based scoring or specialized Cross-Encoder models, reranking allows for more nuanced and accurate assessment of document relevance. This improved relevance translates directly to better performance in downstream tasks, making reranking an essential component in advanced RAG implementations.

The choice between LLM-based and Cross-Encoder reranking methods depends on factors such as required accuracy, available computational resources, and specific application needs. Both approaches offer substantial improvements over basic retrieval methods and contribute to the overall effectiveness of RAG systems.

<div style="text-align: center;">

<img src="../images/reranking-visualization.svg" alt="rerank llm" style="width:100%; height:auto;">
</div>

<div style="text-align: center;">

<img src="../images/reranking_comparison.svg" alt="rerank llm" style="width:100%; height:auto;">
</div>

# Package Installation and Imports

The cell below installs all necessary packages required to run this notebook.
"""
logger.info("# Reranking Methods in RAG Systems")

# !pip install langchain langchain-openai python-dotenv sentence-transformers

# !git clone https://github.com/NirDiamant/RAG_TECHNIQUES.git
sys.path.append('RAG_TECHNIQUES')




load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

"""
### Define the document's path
"""
logger.info("### Define the document's path")

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
## Method 1: LLM based function to rerank the retrieved documents

<div style="text-align: center;">

<img src="../images/rerank_llm.svg" alt="rerank llm" style="width:40%; height:auto;">
</div>

### Create a custom reranking function
"""
logger.info("## Method 1: LLM based function to rerank the retrieved documents")

class RatingScore(BaseModel):
    relevance_score: float = Field(..., description="The relevance score of a document to a query.")

def rerank_documents(query: str, docs: List[Document], top_n: int = 3) -> List[Document]:
    prompt_template = PromptTemplate(
        input_variables=["query", "doc"],
        template="""On a scale of 1-10, rate the relevance of the following document to the query. Consider the specific context and intent of the query, not just keyword matches.
        Query: {query}
        Document: {doc}
        Relevance Score:"""
    )

    llm = ChatOllama(model="llama3.1")
    llm_chain = prompt_template | llm.with_structured_output(RatingScore)

    scored_docs = []
    for doc in docs:
        input_data = {"query": query, "doc": doc.page_content}
        score = llm_chain.invoke(input_data).relevance_score
        try:
            score = float(score)
        except ValueError:
            score = 0  # Default score if parsing fails
        scored_docs.append((doc, score))

    reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked_docs[:top_n]]

"""
### Example usage of the reranking function with a sample query relevant to the document
"""
logger.info("### Example usage of the reranking function with a sample query relevant to the document")

query = "What are the impacts of climate change on biodiversity?"
initial_docs = vectorstore.similarity_search(query, k=15)
reranked_docs = rerank_documents(query, initial_docs)

logger.debug("Top initial documents:")
for i, doc in enumerate(initial_docs[:3]):
    logger.debug(f"\nDocument {i+1}:")
    logger.debug(doc.page_content[:200] + "...")  # Print first 200 characters of each document


logger.debug(f"Query: {query}\n")
logger.debug("Top reranked documents:")
for i, doc in enumerate(reranked_docs):
    logger.debug(f"\nDocument {i+1}:")
    logger.debug(doc.page_content[:200] + "...")  # Print first 200 characters of each document

"""
### Create a custom retriever based on our reranker
"""
logger.info("### Create a custom retriever based on our reranker")

class CustomRetriever(BaseRetriever, BaseModel):

    vectorstore: Any = Field(description="Vector store for initial retrieval")

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str, num_docs=2) -> List[Document]:
        initial_docs = self.vectorstore.similarity_search(query, k=30)
        return rerank_documents(query, initial_docs, top_n=num_docs)


custom_retriever = CustomRetriever(vectorstore=vectorstore)

llm = ChatOllama(model="llama3.1")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=custom_retriever,
    return_source_documents=True
)

"""
### Example query
"""
logger.info("### Example query")

result = qa_chain({"query": query})

logger.debug(f"\nQuestion: {query}")
logger.debug(f"Answer: {result['result']}")
logger.debug("\nRelevant source documents:")
for i, doc in enumerate(result["source_documents"]):
    logger.debug(f"\nDocument {i+1}:")
    logger.debug(doc.page_content[:200] + "...")  # Print first 200 characters of each document

"""
### Example that demonstrates why we should use reranking
"""
logger.info("### Example that demonstrates why we should use reranking")

chunks = [
    "The capital of France is great.",
    "The capital of France is huge.",
    "The capital of France is beautiful.",
    """Have you ever visited Paris? It is a beautiful city where you can eat delicious food and see the Eiffel Tower.
    I really enjoyed all the cities in france, but its capital with the Eiffel Tower is my favorite city.""",
    "I really enjoyed my trip to Paris, France. The city is beautiful and the food is delicious. I would love to visit again. Such a great capital city."
]
docs = [Document(page_content=sentence) for sentence in chunks]


def compare_rag_techniques(query: str, docs: List[Document] = docs) -> None:
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vectorstore = FAISS.from_documents(docs, embeddings)

    logger.debug("Comparison of Retrieval Techniques")
    logger.debug("==================================")
    logger.debug(f"Query: {query}\n")

    logger.debug("Baseline Retrieval Result:")
    baseline_docs = vectorstore.similarity_search(query, k=2)
    for i, doc in enumerate(baseline_docs):
        logger.debug(f"\nDocument {i+1}:")
        logger.debug(doc.page_content)

    logger.debug("\nAdvanced Retrieval Result:")
    custom_retriever = CustomRetriever(vectorstore=vectorstore)
    advanced_docs = custom_retriever.get_relevant_documents(query)
    for i, doc in enumerate(advanced_docs):
        logger.debug(f"\nDocument {i+1}:")
        logger.debug(doc.page_content)


query = "what is the capital of france?"
compare_rag_techniques(query, docs)

"""
## Method 2: Cross Encoder models

<div style="text-align: center;">

<img src="../images/rerank_cross_encoder.svg" alt="rerank cross encoder" style="width:40%; height:auto;">
</div>

### Define the cross encoder class
"""
logger.info("## Method 2: Cross Encoder models")

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

class CrossEncoderRetriever(BaseRetriever, BaseModel):
    vectorstore: Any = Field(description="Vector store for initial retrieval")
    cross_encoder: Any = Field(description="Cross-encoder model for reranking")
    k: int = Field(default=5, description="Number of documents to retrieve initially")
    rerank_top_k: int = Field(default=3, description="Number of documents to return after reranking")

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        initial_docs = self.vectorstore.similarity_search(query, k=self.k)

        pairs = [[query, doc.page_content] for doc in initial_docs]

        scores = self.cross_encoder.predict(pairs)

        scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in scored_docs[:self.rerank_top_k]]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("Async retrieval not implemented")

"""
### Create an instance and showcase over an example
"""
logger.info("### Create an instance and showcase over an example")

cross_encoder_retriever = CrossEncoderRetriever(
    vectorstore=vectorstore,
    cross_encoder=cross_encoder,
    k=10,  # Retrieve 10 documents initially
    rerank_top_k=5  # Return top 5 after reranking
)

llm = ChatOllama(model="llama3.1")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=cross_encoder_retriever,
    return_source_documents=True
)

query = "What are the impacts of climate change on biodiversity?"
result = qa_chain({"query": query})

logger.debug(f"\nQuestion: {query}")
logger.debug(f"Answer: {result['result']}")
logger.debug("\nRelevant source documents:")
for i, doc in enumerate(result["source_documents"]):
    logger.debug(f"\nDocument {i+1}:")
    logger.debug(doc.page_content[:200] + "...")  # Print first 200 characters of each document

logger.info("\n\n[DONE]", bright=True)