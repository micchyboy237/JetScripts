from jet.adapters.langchain.chat_ollama import ChatOllama, OllamaEmbeddings
from jet.logger import logger
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import DeepLake
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
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
# QA using Activeloop's DeepLake
In this tutorial, we are going to use Langchain + Activeloop's Deep Lake with GPT4 to semantically search and ask questions over a group chat.

View a working demo [here](https://twitter.com/thisissukh_/status/1647223328363679745)

## 1. Install required packages
"""
logger.info("# QA using Activeloop's DeepLake")

# !python3 -m pip install --upgrade langchain 'deeplake[enterprise]' ollama tiktoken

"""
## 2. Add API keys


"""
logger.info("## 2. Add API keys")

# import getpass


# os.environ["OPENAI_API_KEY"] = getpass.getpass("Ollama API Key:")
# activeloop_token = getpass.getpass("Activeloop Token:")
os.environ["ACTIVELOOP_TOKEN"] = activeloop_token
# os.environ["ACTIVELOOP_ORG"] = getpass.getpass("Activeloop Org:")

org_id = os.environ["ACTIVELOOP_ORG"]
embeddings = OllamaEmbeddings(model="nomic-embed-text")

dataset_path = "hub://" + org_id + "/data"

"""
## 2. Create sample data

You can generate a sample group chat conversation using ChatGPT with this prompt:

```
Generate a group chat conversation with three friends talking about their day, referencing real places and fictional names. Make it funny and as detailed as possible.
```

I've already generated such a chat in `messages.txt`. We can keep it simple and use this for our example.

## 3. Ingest chat embeddings

We load the messages in the text file, chunk and upload to ActiveLoop Vector store.
"""
logger.info("## 2. Create sample data")

with open("messages.txt") as f:
    state_of_the_union = f.read()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
pages = text_splitter.split_text(state_of_the_union)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100)
texts = text_splitter.create_documents(pages)

logger.debug(texts)

dataset_path = "hub://" + org_id + "/data"
embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = DeepLake.from_documents(
    texts, embeddings, dataset_path=dataset_path, overwrite=True
)

"""
`Optional`: You can also use Deep Lake's Managed Tensor Database as a hosting service and run queries there. In order to do so, it is necessary to specify the runtime parameter as {'tensor_db': True} during the creation of the vector store. This configuration enables the execution of queries on the Managed Tensor Database, rather than on the client side. It should be noted that this functionality is not applicable to datasets stored locally or in-memory. In the event that a vector store has already been created outside of the Managed Tensor Database, it is possible to transfer it to the Managed Tensor Database by following the prescribed steps.
"""


"""
## 4. Ask questions

Now we can ask a question and get an answer back with a semantic search:
"""
logger.info("## 4. Ask questions")

db = DeepLake(dataset_path=dataset_path, read_only=True, embedding=embeddings)

retriever = db.as_retriever()
retriever.search_kwargs["distance_metric"] = "cos"
retriever.search_kwargs["k"] = 4

qa = RetrievalQA.from_chain_type(
    llm=Ollama(), chain_type="stuff", retriever=retriever, return_source_documents=False
)

query = input("Enter query:")

ans = qa({"query": query})

logger.debug(ans)

logger.info("\n\n[DONE]", bright=True)
