from jet.logger import logger
from langchain_pinecone import PineconeEmbeddings
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
# Pinecone Embeddings

Pinecone's inference API can be accessed via `PineconeEmbeddings`. Providing text embeddings via the Pinecone service. We start by installing prerequisite libraries:
"""
logger.info("# Pinecone Embeddings")

# !pip install -qU "langchain-pinecone>=0.2.0"

"""
Next, we [sign up / log in to Pinecone](https://app.pinecone.io) to get our API key:
"""
logger.info("Next, we [sign up / log in to Pinecone](https://app.pinecone.io) to get our API key:")

# from getpass import getpass

# os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY") or getpass(
    "Enter your Pinecone API key: "
)

"""
Check the document for available [models](https://docs.pinecone.io/models/overview). Now we initialize our embedding model like so:
"""
logger.info("Check the document for available [models](https://docs.pinecone.io/models/overview). Now we initialize our embedding model like so:")


embeddings = PineconeEmbeddings(model="multilingual-e5-large")

"""
From here we can create embeddings either sync or async, let's start with sync! We embed a single text as a query embedding (ie what we search with in RAG) using `embed_query`:
"""
logger.info("From here we can create embeddings either sync or async, let's start with sync! We embed a single text as a query embedding (ie what we search with in RAG) using `embed_query`:")

docs = [
    "Apple is a popular fruit known for its sweetness and crisp texture.",
    "The tech company Apple is known for its innovative products like the iPhone.",
    "Many people enjoy eating apples as a healthy snack.",
    "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces.",
    "An apple a day keeps the doctor away, as the saying goes.",
]

doc_embeds = embeddings.embed_documents(docs)
doc_embeds

query = "Tell me about the tech company known as Apple"
query_embed = embeddings.embed_query(query)
query_embed

logger.info("\n\n[DONE]", bright=True)