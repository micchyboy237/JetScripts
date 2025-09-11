from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Epsilla
from langchain_text_splitters import CharacterTextSplitter
from pyepsilla import vectordb
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
# Epsilla

>[Epsilla](https://www.epsilla.com) is an open-source vector database that leverages the advanced parallel graph traversal techniques for vector indexing. Epsilla is licensed under GPL-3.0.

You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration

This notebook shows how to use the functionalities related to the `Epsilla` vector database.

As a prerequisite, you need to have a running Epsilla vector database (for example, through our docker image), and install the ``pyepsilla`` package. View full docs at [docs](https://epsilla-inc.gitbook.io/epsilladb/quick-start).
"""
logger.info("# Epsilla")

# !pip/pip3 install pyepsilla

"""
We want to use OllamaEmbeddings so we have to get the Ollama API Key.
"""
logger.info("We want to use OllamaEmbeddings so we have to get the Ollama API Key.")

# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:")

"""
Ollama API Key: ········
"""
logger.info("Ollama API Key: ········")



loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()

documents = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0).split_documents(
    documents
)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

"""
Epsilla vectordb is running with default host "localhost" and port "8888". We have a custom db path, db name and collection name instead of the default ones.
"""
logger.info("Epsilla vectordb is running with default host "localhost" and port "8888". We have a custom db path, db name and collection name instead of the default ones.")


client = vectordb.Client()
vector_store = Epsilla.from_documents(
    documents,
    embeddings,
    client,
    db_path="/tmp/mypath",
    db_name="MyDB",
    collection_name="MyCollection",
)

query = "What did the president say about Ketanji Brown Jackson"
docs = vector_store.similarity_search(query)
logger.debug(docs[0].page_content)

"""
In state after state, new laws have been passed, not only to suppress the vote, but to subvert entire elections.

We cannot let this happen.

Tonight. I call on the Senate to: Pass the Freedom to Vote Act. Pass the John Lewis Voting Rights Act. And while you’re at it, pass the Disclose Act so Americans can know who is funding our elections.

Tonight, I’d like to honor someone who has dedicated his life to serve this country: Justice Stephen Breyer—an Army veteran, Constitutional scholar, and retiring Justice of the United States Supreme Court. Justice Breyer, thank you for your service.

One of the most serious constitutional responsibilities a President has is nominating someone to serve on the United States Supreme Court.

And I did that 4 days ago, when I nominated Circuit Court of Appeals Judge Ketanji Brown Jackson. One of our nation’s top legal minds, who will continue Justice Breyer’s legacy of excellence.
"""
logger.info("In state after state, new laws have been passed, not only to suppress the vote, but to subvert entire elections.")

logger.info("\n\n[DONE]", bright=True)