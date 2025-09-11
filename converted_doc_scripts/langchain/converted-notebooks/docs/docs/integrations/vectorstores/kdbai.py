from jet.adapters.langchain.chat_ollama import ChatOllama, OllamaEmbeddings
from jet.logger import logger
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import KDBAI
import kdbai_client as kdbai
import os
import pandas as pd
import requests
import shutil
import time


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
# KDB.AI

> [KDB.AI](https://kdb.ai/) is a powerful knowledge-based vector database and search engine that allows you to build scalable, reliable AI applications, using real-time data, by providing advanced search, recommendation and personalization.

[This example](https://github.com/KxSystems/kdbai-samples/blob/main/document_search/document_search.ipynb) demonstrates how to use KDB.AI to run semantic search on unstructured text documents.

To access your end point and API keys, [sign up to KDB.AI here](https://kdb.ai/get-started/).

To set up your development environment, follow the instructions on the [KDB.AI pre-requisites page](https://code.kx.com/kdbai/pre-requisites.html).

The following examples demonstrate some of the ways you can interact with KDB.AI through LangChain.

You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration

## Import required packages
"""
logger.info("# KDB.AI")

# from getpass import getpass


KDBAI_ENDPOINT = input("KDB.AI endpoint: ")
# KDBAI_API_KEY = getpass("KDB.AI API key: ")
# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass("Ollama API Key: ")

TEMP = 0.0
K = 3

"""
## Create a KBD.AI Session
"""
logger.info("## Create a KBD.AI Session")

logger.debug("Create a KDB.AI session...")
session = kdbai.Session(endpoint=KDBAI_ENDPOINT, api_key=KDBAI_API_KEY)

"""
## Create a table
"""
logger.info("## Create a table")

logger.debug('Create table "documents"...')
schema = {
    "columns": [
        {"name": "id", "pytype": "str"},
        {"name": "text", "pytype": "bytes"},
        {
            "name": "embeddings",
            "pytype": "float32",
            "vectorIndex": {"dims": 1536, "metric": "L2", "type": "hnsw"},
        },
        {"name": "tag", "pytype": "str"},
        {"name": "title", "pytype": "bytes"},
    ]
}
table = session.create_table("documents", schema)

# %%time
URL = "https://www.conseil-constitutionnel.fr/node/3850/pdf"
PDF = "Déclaration_des_droits_de_l_homme_et_du_citoyen.pdf"
open(PDF, "wb").write(requests.get(URL).content)

"""
## Read a PDF
"""
logger.info("## Read a PDF")

# %%time
logger.debug("Read a PDF...")
loader = PyPDFLoader(PDF)
pages = loader.load_and_split()
len(pages)

"""
## Create a Vector Database from PDF Text
"""
logger.info("## Create a Vector Database from PDF Text")

# %%time
logger.debug("Create a Vector Database from PDF text...")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
texts = [p.page_content for p in pages]
metadata = pd.DataFrame(index=list(range(len(texts))))
metadata["tag"] = "law"
metadata["title"] = "Déclaration des Droits de l'Homme et du Citoyen de 1789".encode(
    "utf-8"
)
vectordb = KDBAI(table, embeddings)
vectordb.add_texts(texts=texts, metadatas=metadata)

"""
## Create LangChain Pipeline
"""
logger.info("## Create LangChain Pipeline")

# %%time
logger.debug("Create LangChain Pipeline...")
qabot = RetrievalQA.from_chain_type(
    chain_type="stuff",
    llm=ChatOllama(model="llama3.2"),
    retriever=vectordb.as_retriever(search_kwargs=dict(k=K)),
    return_source_documents=True,
)

"""
## Summarize the document in English
"""
logger.info("## Summarize the document in English")

# %%time
Q = "Summarize the document in English:"
logger.debug(f"\n\n{Q}\n")
logger.debug(qabot.invoke(dict(query=Q))["result"])

"""
## Query the Data
"""
logger.info("## Query the Data")

# %%time
Q = "Is it a fair law and why ?"
logger.debug(f"\n\n{Q}\n")
logger.debug(qabot.invoke(dict(query=Q))["result"])

# %%time
Q = "What are the rights and duties of the man, the citizen and the society ?"
logger.debug(f"\n\n{Q}\n")
logger.debug(qabot.invoke(dict(query=Q))["result"])

# %%time
Q = "Is this law practical ?"
logger.debug(f"\n\n{Q}\n")
logger.debug(qabot.invoke(dict(query=Q))["result"])

"""
## Clean up the Documents table
"""
logger.info("## Clean up the Documents table")

session.table("documents").drop()

logger.info("\n\n[DONE]", bright=True)