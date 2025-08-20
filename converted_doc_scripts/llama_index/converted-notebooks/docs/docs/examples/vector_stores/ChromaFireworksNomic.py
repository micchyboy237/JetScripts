from IPython.display import Markdown, display
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.embeddings.fireworks import FireworksEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.fireworks import Fireworks
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/ChromaIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Chroma + Fireworks + Nomic with Matryoshka embedding

This example is adapted from the ChromaIndex example, + how to use Matryoshka embedding from Nomic on top of Fireworks.ai.

## Chroma

>[Chroma](https://docs.trychroma.com/getting-started) is a AI-native open-source vector database focused on developer productivity and happiness. Chroma is licensed under Apache 2.0.

<a href="https://discord.gg/MMeYNTmh3x" target="_blank">
      <img src="https://img.shields.io/discord/1073293645303795742" alt="Discord">
  </a>&nbsp;&nbsp;
  <a href="https://github.com/chroma-core/chroma/blob/master/LICENSE" target="_blank">
      <img src="https://img.shields.io/static/v1?label=license&message=Apache 2.0&color=white" alt="License">
  </a>&nbsp;&nbsp;
  <img src="https://github.com/chroma-core/chroma/actions/workflows/chroma-integration-test.yml/badge.svg?branch=main" alt="Integration Tests">

- [Website](https://www.trychroma.com/)
- [Documentation](https://docs.trychroma.com/)
- [Twitter](https://twitter.com/trychroma)
- [Discord](https://discord.gg/MMeYNTmh3x)

Chroma is fully-typed, fully-tested and fully-documented.

Install Chroma with:

```sh
pip install chromadb
```

Chroma runs in various modes. See below for examples of each integrated with LangChain.
- `in-memory` - in a python script or jupyter notebook
- `in-memory with persistence` - in a script or notebook and save/load to disk
- `in a docker container` - as a server running your local machine or in the cloud

Like any other database, you can: 
- `.add` 
- `.get` 
- `.update`
- `.upsert`
- `.delete`
- `.peek`
- and `.query` runs the similarity search.

View full docs at [docs](https://docs.trychroma.com/reference/Collection). 

## Nomic

Nomic published a new embedding model nomic-ai/nomic-embed-text-v1.5 that is capable of returning variable embedding size depending on how cost sensitive you are. For more information please check out their model [here](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) and their website [here](https://home.nomic.ai/)

## Fireworks.ai

Fireworks is the leading OSS model inference provider. In this example we will use Fireworks to run the nomic model as well as mixtral-8x7b-instruct model as the query engine. For more information about fireworks, please check out their website [here](https://fireworks.ai)

## Basic Example

In this basic example, we take the Paul Graham essay, split it into chunks, embed it using an open-source embedding model, load it into Chroma, and then query it.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Chroma + Fireworks + Nomic with Matryoshka embedding")

# %pip install -q llama-index-vector-stores-chroma llama-index-llms-fireworks llama-index-embeddings-fireworks==0.1.2

# %pip install -q llama-index

"""
#### Creating a Chroma Index
"""
logger.info("#### Creating a Chroma Index")

# !pip install llama-index chromadb --quiet
# !pip install -q chromadb
# !pip install -q pydantic==1.10.11


# import getpass

# fw_api_key = getpass.getpass("Fireworks API Key:")

"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'


llm = Fireworks(
    temperature=0, model="accounts/fireworks/models/mixtral-8x7b-instruct"
)

chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("quickstart")

embed_model = FireworksEmbedding(
    model_name="nomic-ai/nomic-embed-text-v1.5",
)

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("What did the author do growing up?")
display(Markdown(f"<b>{response}</b>"))

"""
## Basic Example (including saving to disk) and resizable embeddings

Extending the previous example, if you want to save to disk, simply initialize the Chroma client and pass the directory where you want the data to be saved to. 

`Caution`: Chroma makes a best-effort to automatically save data to disk, however multiple in-memory clients can stomp each other's work. As a best practice, only have one client per path running at any given time.

Also we are going to resize the embeddings down to 128 dimensions. This is helpful for cases where you are cost conscious on the database side.
"""
logger.info("## Basic Example (including saving to disk) and resizable embeddings")

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

embed_model = FireworksEmbedding(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    api_base="https://api.fireworks.ai/inference/v1",
    dimensions=128,
)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

db2 = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db2.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
)

query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("What did the author do growing up?")
display(Markdown(f"<b>{response}</b>"))

"""
You can see that the results are the same across the two, so you can experiment with the dimension sizes and then experiment with the cost and quality trade-off yourself.
"""
logger.info("You can see that the results are the same across the two, so you can experiment with the dimension sizes and then experiment with the cost and quality trade-off yourself.")

logger.info("\n\n[DONE]", bright=True)