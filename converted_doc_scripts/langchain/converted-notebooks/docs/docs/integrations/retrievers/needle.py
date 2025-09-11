from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders.needle import NeedleLoader
from langchain_community.retrievers.needle import NeedleRetriever
from langchain_core.prompts import ChatPromptTemplate
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
## Needle Retriever
[Needle](https://needle-ai.com) makes it easy to create your RAG pipelines with minimal effort. 

For more details, refer to our [API documentation](https://docs.needle-ai.com/docs/api-reference/needle-api)

## Overview
The Needle Document Loader is a utility for integrating Needle collections with LangChain. It enables seamless storage, retrieval, and utilization of documents for Retrieval-Augmented Generation (RAG) workflows.

This example demonstrates:

* Storing documents into a Needle collection.
* Setting up a retriever to fetch documents.
* Building a Retrieval-Augmented Generation (RAG) pipeline.

### Setup
Before starting, ensure you have the following environment variables set:

* NEEDLE_API_KEY: Your API key for authenticating with Needle.
# * OPENAI_API_KEY: Your Ollama API key for language model operations.

## Initialization
To initialize the NeedleLoader, you need the following parameters:

* needle_api_key: Your Needle API key (or set it as an environment variable).
* collection_id: The ID of the Needle collection to work with.
"""
logger.info("## Needle Retriever")


os.environ["NEEDLE_API_KEY"] = ""

# os.environ["OPENAI_API_KEY"] = ""

"""
## Instantiation
"""
logger.info("## Instantiation")


collection_id = "clt_01J87M9T6B71DHZTHNXYZQRG5H"

document_loader = NeedleLoader(
    needle_api_key=os.getenv("NEEDLE_API_KEY"),
    collection_id=collection_id,
)

"""
## Load
To add files to the Needle collection:
"""
logger.info("## Load")

files = {
    "tech-radar-30.pdf": "https://www.thoughtworks.com/content/dam/thoughtworks/documents/radar/2024/04/tr_technology_radar_vol_30_en.pdf"
}

document_loader.add_files(files=files)


"""
## Usage
### Use within a chain
Below is a complete example of setting up a RAG pipeline with Needle within a chain:
"""
logger.info("## Usage")



llm = ChatOllama(model="llama3.2")

retriever = NeedleRetriever(
    needle_api_key=os.getenv("NEEDLE_API_KEY"),
    collection_id="clt_01J87M9T6B71DHZTHNXYZQRG5H",
)

system_prompt = """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know, say so concisely.\n\n{context}
    """

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{input}")]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)

rag_chain = create_retrieval_chain(retriever, question_answer_chain)

query = {"input": "Did RAG move to accepted?"}

response = rag_chain.invoke(query)

response

"""
## API reference

For detailed documentation of all `Needle` features and configurations head to the API reference: https://docs.needle-ai.com
"""
logger.info("## API reference")

logger.info("\n\n[DONE]", bright=True)