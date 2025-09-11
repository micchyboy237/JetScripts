from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Typesense
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
# Typesense

> [Typesense](https://typesense.org) is an open-source, in-memory search engine, that you can either [self-host](https://typesense.org/docs/guide/install-typesense#option-2-local-machine-self-hosting) or run on [Typesense Cloud](https://cloud.typesense.org/).
>
> Typesense focuses on performance by storing the entire index in RAM (with a backup on disk) and also focuses on providing an out-of-the-box developer experience by simplifying available options and setting good defaults.
>
> It also lets you combine attribute-based filtering together with vector queries, to fetch the most relevant documents.

This notebook shows you how to use Typesense as your VectorStore.

Let's first install our dependencies:
"""
logger.info("# Typesense")

# %pip install --upgrade --quiet  typesense openapi-schema-pydantic langchain-ollama langchain-community tiktoken

"""
We want to use `OllamaEmbeddings` so we have to get the Ollama API Key.
"""
logger.info("We want to use `OllamaEmbeddings` so we have to get the Ollama API Key.")

# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:")


"""
Let's import our test dataset:
"""
logger.info("Let's import our test dataset:")

loader = TextLoader("../../how_to/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

docsearch = Typesense.from_documents(
    docs,
    embeddings,
    typesense_client_params={
        "host": "localhost",  # Use xxx.a1.typesense.net for Typesense Cloud
        "port": "8108",  # Use 443 for Typesense Cloud
        "protocol": "http",  # Use https for Typesense Cloud
        "typesense_api_key": "xyz",
        "typesense_collection_name": "lang-chain",
    },
)

"""
## Similarity Search
"""
logger.info("## Similarity Search")

query = "What did the president say about Ketanji Brown Jackson"
found_docs = docsearch.similarity_search(query)

logger.debug(found_docs[0].page_content)

"""
## Typesense as a Retriever

Typesense, as all the other vector stores, is a LangChain Retriever, by using cosine similarity.
"""
logger.info("## Typesense as a Retriever")

retriever = docsearch.as_retriever()
retriever

query = "What did the president say about Ketanji Brown Jackson"
retriever.invoke(query)[0]

logger.info("\n\n[DONE]", bright=True)