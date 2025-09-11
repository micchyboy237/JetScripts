from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain import hub
from langchain.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# RAG Fusion

Re-implemented from [this GitHub repo](https://github.com/Raudaschl/rag-fusion), all credit to original author

> RAG-Fusion, a search methodology that aims to bridge the gap between traditional search paradigms and the multifaceted dimensions of human queries. Inspired by the capabilities of Retrieval Augmented Generation (RAG), this project goes a step further by employing multiple query generation and Reciprocal Rank Fusion to re-rank search results.

## Setup

For this example, we will use Pinecone and some fake data. To configure Pinecone, set the following environment variable:

- `PINECONE_API_KEY`: Your Pinecone API key
"""
logger.info("# RAG Fusion")


all_documents = {
    "doc1": "Climate change and economic impact.",
    "doc2": "Public health concerns due to climate change.",
    "doc3": "Climate change: A social perspective.",
    "doc4": "Technological solutions to climate change.",
    "doc5": "Policy changes needed to combat climate change.",
    "doc6": "Climate change and its impact on biodiversity.",
    "doc7": "Climate change: The science and models.",
    "doc8": "Global warming: A subset of climate change.",
    "doc9": "How climate change affects daily weather.",
    "doc10": "The history of climate change activism.",
}

vectorstore = PineconeVectorStore.from_texts(
    list(all_documents.values()), OllamaEmbeddings(model="mxbai-embed-large"), index_name="rag-fusion"
)

"""
## Define the Query Generator

We will now define a chain to do the query generation
"""
logger.info("## Define the Query Generator")



prompt = hub.pull("langchain-ai/rag-fusion-query-generation")



generate_queries = (
    prompt | ChatOllama(model="llama3.2") | StrOutputParser() | (lambda x: x.split("\n"))
)

"""
## Define the full chain

We can now put it all together and define the full chain. This chain:
    
    1. Generates a bunch of queries
    2. Looks up each query in the retriever
    3. Joins all the results together using reciprocal rank fusion
    
    
Note that it does NOT do a final generation step
"""
logger.info("## Define the full chain")

original_query = "impact of climate change"

vectorstore = PineconeVectorStore.from_existing_index("rag-fusion", OllamaEmbeddings(model="mxbai-embed-large"))
retriever = vectorstore.as_retriever()



def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results

chain = generate_queries | retriever.map() | reciprocal_rank_fusion

chain.invoke({"original_query": original_query})

logger.info("\n\n[DONE]", bright=True)