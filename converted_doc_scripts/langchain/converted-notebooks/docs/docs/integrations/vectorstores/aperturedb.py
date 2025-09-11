from jet.logger import logger
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import ApertureDB
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
# ApertureDB

[ApertureDB](https://docs.aperturedata.io) is a database that stores, indexes, and manages multi-modal data like text, images, videos, bounding boxes, and embeddings, together with their associated metadata.

This notebook explains how to use the embeddings functionality of ApertureDB.

## Install ApertureDB Python SDK

This installs the [Python SDK](https://docs.aperturedata.io/category/aperturedb-python-sdk) used to write client code for ApertureDB.
"""
logger.info("# ApertureDB")

# %pip install --upgrade --quiet aperturedb

"""
## Run an ApertureDB instance

To continue, you should have an [ApertureDB instance up and running](https://docs.aperturedata.io/HowToGuides/start/Setup) and configure your environment to use it.  
There are various ways to do that, for example:

```bash
docker run --publish 55555:55555 aperturedata/aperturedb-standalone
adb config create local --active --no-interactive
```

## Download some web documents
We're going to do a mini-crawl here of one web page.
"""
logger.info("## Run an ApertureDB instance")


loader = WebBaseLoader("https://docs.aperturedata.io")
docs = loader.load()

"""
## Select embeddings model

We want to use OllamaEmbeddings so we have to import the necessary modules.

Ollama can be set up as a docker container as described in the [documentation](https://hub.docker.com/r/ollama/ollama), for example:
```bash
# Run server
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
# Tell server to load a specific model
docker exec ollama ollama run llama2
```
"""
logger.info("## Select embeddings model")


embeddings = OllamaEmbeddings(model="mxbai-embed-large")

"""
## Split documents into segments

We want to turn our single document into multiple segments.
"""
logger.info("## Split documents into segments")


text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

"""
## Create vectorstore from documents and embeddings

This code creates a vectorstore in the ApertureDB instance.
Within the instance, this vectorstore is represented as a "[descriptor set](https://docs.aperturedata.io/category/descriptorset-commands)".
By default, the descriptor set is named `langchain`.  The following code will generate embeddings for each document and store them in ApertureDB as descriptors.  This will take a few seconds as the embeddings are bring generated.
"""
logger.info("## Create vectorstore from documents and embeddings")


vector_db = ApertureDB.from_documents(documents, embeddings)

"""
## Select a large language model

Again, we use the Ollama server we set up for local processing.
"""
logger.info("## Select a large language model")


llm = ChatOllama(model="llama2")

"""
## Build a RAG chain

Now we have all the components we need to create a RAG (Retrieval-Augmented Generation) chain.  This chain does the following:
1. Generate embedding descriptor for user query
2. Find text segments that are similar to the user query using the vector store
3. Pass user query and context documents to the LLM using a prompt template
4. Return the LLM's answer
"""
logger.info("## Build a RAG chain")


prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")


document_chain = create_stuff_documents_chain(llm, prompt)


retriever = vector_db.as_retriever()


retrieval_chain = create_retrieval_chain(retriever, document_chain)

"""
## Run the RAG chain

Finally we pass a question to the chain and get our answer.  This will take a few seconds to run as the LLM generates an answer from the query and context documents.
"""
logger.info("## Run the RAG chain")

user_query = "How can ApertureDB store images?"
response = retrieval_chain.invoke({"input": user_query})
logger.debug(response["answer"])

logger.info("\n\n[DONE]", bright=True)
