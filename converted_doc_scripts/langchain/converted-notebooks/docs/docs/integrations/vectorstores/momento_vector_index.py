from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.ollama_embeddings import OllamaEmbeddings
from jet.logger import logger
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import MomentoVectorIndex
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
# Momento Vector Index (MVI)

>[MVI](https://gomomento.com): the most productive, easiest to use, serverless vector index for your data. To get started with MVI, simply sign up for an account. There's no need to handle infrastructure, manage servers, or be concerned about scaling. MVI is a service that scales automatically to meet your needs.

To sign up and access MVI, visit the [Momento Console](https://console.gomomento.com).

# Setup

## Install prerequisites

You will need:
- the [`momento`](https://pypi.org/project/momento/) package for interacting with MVI, and
- the ollama package for interacting with the Ollama API.
- the tiktoken package for tokenizing text.
"""
logger.info("# Momento Vector Index (MVI)")

# %pip install --upgrade --quiet  momento langchain-ollama langchain-community tiktoken

"""
## Enter API keys
"""
logger.info("## Enter API keys")

# import getpass

"""
### Momento: for indexing data

Visit the [Momento Console](https://console.gomomento.com) to get your API key.
"""
logger.info("### Momento: for indexing data")

if "MOMENTO_API_KEY" not in os.environ:
    #     os.environ["MOMENTO_API_KEY"] = getpass.getpass("Momento API Key:")

"""
### Ollama: for text embeddings
"""
logger.info("### Ollama: for text embeddings")

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:")

"""
# Load your data

Here we use the example dataset from Langchain, the state of the union address.

First we load relevant modules:
"""
logger.info("# Load your data")


"""
Then we load the data:
"""
logger.info("Then we load the data:")

loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
len(documents)

"""
Note the data is one large file, hence there is only one document:
"""
logger.info("Note the data is one large file, hence there is only one document:")

len(documents[0].page_content)

"""
Because this is one large text file, we split it into chunks for question answering. That way, user questions will be answered from the most relevant chunk.
"""
logger.info("Because this is one large text file, we split it into chunks for question answering. That way, user questions will be answered from the most relevant chunk.")

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
len(docs)

"""
# Index your data

Indexing your data is as simple as instantiating the `MomentoVectorIndex` object. Here we use the `from_documents` helper to both instantiate and index the data:
"""
logger.info("# Index your data")

vector_db = MomentoVectorIndex.from_documents(
    docs, OllamaEmbeddings(model="nomic-embed-text"), index_name="sotu"
)

"""
This connects to the Momento Vector Index service using your API key and indexes the data. If the index did not exist before, this process creates it for you. The data is now searchable.

# Query your data

## Ask a question directly against the index

The most direct way to query the data is to search against the index. We can do that as follows using the `VectorStore` API:
"""
logger.info("# Query your data")

query = "What did the president say about Ketanji Brown Jackson"
docs = vector_db.similarity_search(query)

docs[0].page_content

"""
While this does contain relevant information about Ketanji Brown Jackson, we don't have a concise, human-readable answer. We'll tackle that in the next section.

## Use an LLM to generate fluent answers

With the data indexed in MVI, we can integrate with any chain that leverages vector similarity search. Here we use the `RetrievalQA` chain to demonstrate how to answer questions from the indexed data.

First we load the relevant modules:
"""
logger.info("## Use an LLM to generate fluent answers")


"""
Then we instantiate the retrieval QA chain:
"""
logger.info("Then we instantiate the retrieval QA chain:")

llm = ChatOllama(model="llama3.2")
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vector_db.as_retriever())

qa_chain({"query": "What did the president say about Ketanji Brown Jackson?"})

"""
# Next Steps

That's it! You've now indexed your data and can query it using the Momento Vector Index. You can use the same index to query your data from any chain that supports vector similarity search.

With Momento you can not only index your vector data, but also cache your API calls and store your chat message history. Check out the other Momento langchain integrations to learn more.

To learn more about the Momento Vector Index, visit the [Momento Documentation](https://docs.gomomento.com).


"""
logger.info("# Next Steps")

logger.info("\n\n[DONE]", bright=True)
