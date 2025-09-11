from jet.adapters.langchain.chat_ollama import ChatOllama, OllamaEmbeddings
from jet.logger import logger
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PsychicLoader
from langchain_text_splitters import CharacterTextSplitter
from psychicapi import ConnectorId
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
# Psychic
This notebook covers how to load documents from `Psychic`. See [here](/docs/integrations/providers/psychic) for more details.

## Prerequisites
1. Follow the Quick Start section in [this document](/docs/integrations/providers/psychic)
2. Log into the [Psychic dashboard](https://dashboard.psychic.dev/) and get your secret key
3. Install the frontend react library into your web app and have a user authenticate a connection. The connection will be created using the connection id that you specify.

## Loading documents

Use the `PsychicLoader` class to load in documents from a connection. Each connection has a connector id (corresponding to the SaaS app that was connected) and a connection id (which you passed in to the frontend library).
"""
logger.info("# Psychic")

# !poetry run pip -q install psychicapi langchain-chroma


google_drive_loader = PsychicLoader(
    connector_id=ConnectorId.gdrive.value,
    connection_id="google-test",
)

documents = google_drive_loader.load()

"""
## Converting the docs to embeddings 

We can now convert these documents into embeddings and store them in a vector database like Chroma
"""
logger.info("## Converting the docs to embeddings")


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
docsearch = Chroma.from_documents(texts, embeddings)
chain = RetrievalQAWithSourcesChain.from_chain_type(
    Ollama(temperature=0), chain_type="stuff", retriever=docsearch.as_retriever()
)
chain({"question": "what is psychic?"}, return_only_outputs=True)

logger.info("\n\n[DONE]", bright=True)
