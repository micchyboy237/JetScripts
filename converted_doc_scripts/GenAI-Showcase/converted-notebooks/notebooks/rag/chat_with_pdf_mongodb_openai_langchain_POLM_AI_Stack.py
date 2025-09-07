from google.colab import userdata
from jet.logger import CustomLogger
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOllama
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import PyPDFLoader
from pymongo import MongoClient
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
## Data Ingestion into MongoDB Database

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/rag/chat_with_pdf_mongodb_ollama_langchain_POLM_AI_Stack.ipynb)

**Steps to creating a MongoDB Database**
- [Register for a free MongoDB Atlas Account](https://www.mongodb.com/cloud/atlas/register?utm_campaign=devrel&utm_source=workshop&utm_medium=organic_social&utm_content=rag%20to%20agents%20notebook&utm_term=richmond.alake)
- [Create a Cluster](https://www.mongodb.com/docs/guides/atlas/cluster/)
- [Get your connection string](https://www.mongodb.com/docs/guides/atlas/connection-string/)

## Vector Index Creation

- [Create an Atlas Vector Search Index](https://www.mongodb.com/docs/compass/current/indexes/create-vector-search-index/)

- If you are following this notebook ensure that you are creating a vector search index for the right database(anthropic_demo) and collection(research)

Below is the vector search index definition for this notebook

```json
{
  "fields": [
    {
      "numDimensions": 1536,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}
```

- Give your vector search index the name "vector_index" if you are following this notebook

## Code
"""
logger.info("## Data Ingestion into MongoDB Database")

# ! pip install --quiet langchain pymongo langchain-ollama langchain-community pypdf



# os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")

mongo_uri = userdata.get("MONGO_URI")
db_name = "anthropic_demo"
collection_name = "research"

client = MongoClient(mongo_uri, appname="devrel.showcase.chat_with_pdf")
db = client[db_name]
collection = db[collection_name]

loader = PyPDFLoader("mapping_llms.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector_store = MongoDBAtlasVectorSearch.from_documents(
    texts, embeddings, collection=collection, index_name="vector_index"
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
llm = ChatOllama(model="llama3.2")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
)


def process_query(query):
    result = qa_chain({"query": query})
    return result["result"], result["source_documents"]


query = "What is the document about?"
answer, sources = process_query(query)
logger.debug(f"Answer: {answer}")
logger.debug("Sources:")
for doc in sources:
    logger.debug(f"- {doc.metadata['source']}: {doc.page_content[:100]}...")

client.close()

logger.info("\n\n[DONE]", bright=True)