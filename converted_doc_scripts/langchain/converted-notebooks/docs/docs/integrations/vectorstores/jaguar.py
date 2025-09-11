from jet.adapters.langchain.chat_ollama import ChatOllama, Ollama, OllamaEmbeddings
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.jaguar import Jaguar
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import CharacterTextSplitter
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
# Jaguar Vector Database

1. It is a distributed vector database
2. The “ZeroMove” feature of JaguarDB enables instant horizontal scalability
3. Multimodal: embeddings, text, images, videos, PDFs, audio, time series, and geospatial
4. All-masters: allows both parallel reads and writes
5. Anomaly detection capabilities
6. RAG support: combines LLM with proprietary and real-time data
7. Shared metadata: sharing of metadata across multiple vector indexes
8. Distance metrics: Euclidean, Cosine, InnerProduct, Manhatten, Chebyshev, Hamming, Jeccard, Minkowski

## Prerequisites

There are two requirements for running the examples in this file.
1. You must install and set up the JaguarDB server and its HTTP gateway server.
   Please refer to the instructions in:
   [www.jaguardb.com](http://www.jaguardb.com)
   For quick setup in docker environment:
   docker pull jaguardb/jaguardb
   docker run -d -p 8888:8888 -p 8080:8080 --name jaguardb  jaguardb/jaguardb

2. You must install the http client package for JaguarDB:
   ```
       pip install -U jaguardb-http-client
   ```
   
3. You'll need to install `langchain-community` with `pip install -qU langchain-community` to use this integration

## RAG With Langchain

This section demonstrates chatting with LLM together with Jaguar in the langchain software stack.
"""
logger.info("# Jaguar Vector Database")


"""
Load a text file into a set of documents
"""
loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
docs = text_splitter.split_documents(documents)

"""
Instantiate a Jaguar vector store
"""
url = "http://192.168.5.88:8080/fwww/"

embeddings = OllamaEmbeddings(model="nomic-embed-text")

pod = "vdb"

store = "langchain_rag_store"

vector_index = "v"

vector_type = "cosine_fraction_float"

vector_dimension = 1536

vectorstore = Jaguar(
    pod, store, vector_index, vector_type, vector_dimension, url, embeddings
)

"""
Login must be performed to authorize the client.
The environment variable JAGUAR_API_KEY or file $HOME/.jagrc
should contain the API key for accessing JaguarDB servers.
"""
vectorstore.login()


"""
Create vector store on the JaguarDB database server.
This should be done only once.
"""
metadata = "category char(16)"

text_size = 4096

vectorstore.create(metadata, text_size)

"""
Add the texts from the text splitter to our vectorstore
"""
vectorstore.add_documents(docs)

""" Get the retriever object """
retriever = vectorstore.as_retriever()

template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

""" Obtain a Large Language Model """
LLM = ChatOllama(model="llama3.2")

""" Create a chain for the RAG flow """
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | LLM
    | StrOutputParser()
)

resp = rag_chain.invoke("What did the president say about Justice Breyer?")
logger.debug(resp)

"""
## Interaction With Jaguar Vector Store

Users can interact directly with the Jaguar vector store for similarity search and anomaly detection.
"""
logger.info("## Interaction With Jaguar Vector Store")


url = "http://192.168.3.88:8080/fwww/"
pod = "vdb"
store = "langchain_test_store"
vector_index = "v"
vector_type = "cosine_fraction_float"
vector_dimension = 10
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Jaguar(
    pod, store, vector_index, vector_type, vector_dimension, url, embeddings
)

vectorstore.login()

metadata_str = "author char(32), category char(16)"
vectorstore.create(metadata_str, 1024)

texts = ["foo", "bar", "baz"]
metadatas = [
    {"author": "Adam", "category": "Music"},
    {"author": "Eve", "category": "Music"},
    {"author": "John", "category": "History"},
]
ids = vectorstore.add_texts(texts=texts, metadatas=metadatas)

output = vectorstore.similarity_search(
    query="foo",
    k=1,
    metadatas=["author", "category"],
)
assert output[0].page_content == "foo"
assert output[0].metadata["author"] == "Adam"
assert output[0].metadata["category"] == "Music"
assert len(output) == 1

where = "author='Eve'"
output = vectorstore.similarity_search(
    query="foo",
    k=3,
    fetch_k=9,
    where=where,
    metadatas=["author", "category"],
)
assert output[0].page_content == "bar"
assert output[0].metadata["author"] == "Eve"
assert output[0].metadata["category"] == "Music"
assert len(output) == 1

result = vectorstore.is_anomalous(
    query="dogs can jump high",
)
assert result is False

vectorstore.clear()
assert vectorstore.count() == 0

vectorstore.drop()

vectorstore.logout()

logger.info("\n\n[DONE]", bright=True)
