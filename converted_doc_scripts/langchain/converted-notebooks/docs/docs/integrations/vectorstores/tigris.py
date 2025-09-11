from jet.adapters.langchain.chat_ollama import OllamaEmbeddings
from jet.logger import logger
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Tigris
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
# Tigris

> [Tigris](https://tigrisdata.com) is an open-source Serverless NoSQL Database and Search Platform designed to simplify building high-performance vector search applications.
> `Tigris` eliminates the infrastructure complexity of managing, operating, and synchronizing multiple tools, allowing you to focus on building great applications instead.

This notebook guides you how to use Tigris as your VectorStore

**Pre requisites**
1. An Ollama account. You can sign up for an account [here](https://platform.ollama.com/)
2. [Sign up for a free Tigris account](https://console.preview.tigrisdata.cloud). Once you have signed up for the Tigris account, create a new project called `vectordemo`. Next, make a note of the *Uri* for the region you've created your project in, the **clientId** and **clientSecret**. You can get all this information from the **Application Keys** section of the project.

Let's first install our dependencies:
"""
logger.info("# Tigris")

# %pip install --upgrade --quiet  tigrisdb openapi-schema-pydantic langchain-ollama langchain-community tiktoken

"""
We will load the `Ollama` api key and `Tigris` credentials in our environment
"""
logger.info("We will load the `Ollama` api key and `Tigris` credentials in our environment")

# import getpass

# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:")
if "TIGRIS_PROJECT" not in os.environ:
#     os.environ["TIGRIS_PROJECT"] = getpass.getpass("Tigris Project Name:")
if "TIGRIS_CLIENT_ID" not in os.environ:
#     os.environ["TIGRIS_CLIENT_ID"] = getpass.getpass("Tigris Client Id:")
if "TIGRIS_CLIENT_SECRET" not in os.environ:
#     os.environ["TIGRIS_CLIENT_SECRET"] = getpass.getpass("Tigris Client Secret:")


"""
### Initialize Tigris vector store
Let's import our test dataset:
"""
logger.info("### Initialize Tigris vector store")

loader = TextLoader("../../../state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

vector_store = Tigris.from_documents(docs, embeddings, index_name="my_embeddings")

"""
### Similarity Search
"""
logger.info("### Similarity Search")

query = "What did the president say about Ketanji Brown Jackson"
found_docs = vector_store.similarity_search(query)
logger.debug(found_docs)

"""
### Similarity Search with score (vector distance)
"""
logger.info("### Similarity Search with score (vector distance)")

query = "What did the president say about Ketanji Brown Jackson"
result = vector_store.similarity_search_with_score(query)
for doc, score in result:
    logger.debug(f"document={doc}, score={score}")

logger.info("\n\n[DONE]", bright=True)