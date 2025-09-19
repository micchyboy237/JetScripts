from IPython.display import Markdown, display
from bagel import Settings
from jet.logger import CustomLogger
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.bagel import BagelVectorStore
import bagel
import openai
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/BagelIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Bagel Network

>[Bagel](https://docs.bageldb.ai/) is a Open Inference Data for AI. It is built for distributed Machine Learning compute. Cutting AI data infra spend by tenfold.

<a href="https://discord.gg/bA7B6r97" target="_blank">
      <img src="https://img.shields.io/discord/1073293645303795742" alt="Discord">
  </a>&nbsp;&nbsp;


- [Website](https://www.bageldb.ai/)
- [Documentation](https://docs.bageldb.ai/)
- [Twitter](https://twitter.com/bageldb_ai)
- [Discord](https://discord.gg/bA7B6r97)


Install Bagel with:

```sh
pip install bagelML
```


Like any other database, you can:
- `.add` 
- `.get` 
- `.delete`
- `.update`
- `.upsert`
- `.peek`
- `.modify`
- and `.find` runs the similarity search.

## Basic Example

In this basic example, we take the a Paul Graham essay, split it into chunks, embed it using an open-source embedding model, load it into Bagel, and then query it.
"""
logger.info("# Bagel Network")

# %pip install llama-index-vector-stores-bagel
# %pip install llama-index-embeddings-huggingface
# %pip install bagelML


# import getpass

# os.environ["OPENAI_API_KEY"] = getpass.getpass("OllamaFunctionCalling API Key:")

# openai.api_key = os.environ["OPENAI_API_KEY"]

"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

server_settings = Settings(
    bagel_api_impl="rest", bagel_server_host="api.bageldb.ai"
)

client = bagel.Client(server_settings)

collection = client.get_or_create_cluster(
    "testing_embeddings", embedding_model="custom", dimension=384
)

embed_model = "local:BAAI/bge-small-en-v1.5"

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

vector_store = BagelVectorStore(collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
logger.debug(f"<b>{response}</b>")

"""
## Create - Add - Get
"""
logger.info("## Create - Add - Get")

def create_add_get(client):
    """
    Create, add, and get
    """
    name = "testing"

    cluster = client.get_or_create_cluster(name)

    resp = cluster.add(
        documents=[
            "This is document1",
            "This is bidhan",
        ],
        metadatas=[{"source": "google"}, {"source": "notion"}],
        ids=[str(uuid.uuid4()), str(uuid.uuid4())],
    )

    logger.debug("count of docs:", cluster.count())

    first_item = cluster.peek(1)
    if first_item:
        logger.debug("get 1st item")

    logger.debug(">> create_add_get done !\n")

"""
## Create - Add - Find by Text
"""
logger.info("## Create - Add - Find by Text")

def create_add_find(client):
    """
    Create, add, & find

    Parameters
    ----------
    api : _type_
        _description_
    """
    name = "testing"

    cluster = client.get_or_create_cluster(name)

    cluster.add(
        documents=[
            "This is document",
            "This is Towhid",
            "This is text",
        ],
        metadatas=[
            {"source": "notion"},
            {"source": "notion"},
            {"source": "google-doc"},
        ],
        ids=[str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())],
    )

    results = cluster.find(
        query_texts=["This"],
        n_results=5,
        where={"source": "notion"},
        where_document={"$contains": "is"},
    )

    logger.debug(results)
    logger.debug(">> create_add_find done  !\n")

"""
## Create - Add - Find by Embeddings
"""
logger.info("## Create - Add - Find by Embeddings")

def create_add_find_em(client):
    """Create, add, & find embeddings

    Parameters
    ----------
    api : _type_
        _description_
    """
    name = "testing_embeddings"
    client.reset()

    cluster = api.get_or_create_cluster(name)
    cluster.add(
        embeddings=[
            [1.1, 2.3, 3.2],
            [4.5, 6.9, 4.4],
            [1.1, 2.3, 3.2],
            [4.5, 6.9, 4.4],
            [1.1, 2.3, 3.2],
            [4.5, 6.9, 4.4],
            [1.1, 2.3, 3.2],
            [4.5, 6.9, 4.4],
        ],
        metadatas=[
            {"uri": "img1.png", "style": "style1"},
            {"uri": "img2.png", "style": "style2"},
            {"uri": "img3.png", "style": "style1"},
            {"uri": "img4.png", "style": "style1"},
            {"uri": "img5.png", "style": "style1"},
            {"uri": "img6.png", "style": "style1"},
            {"uri": "img7.png", "style": "style1"},
            {"uri": "img8.png", "style": "style1"},
        ],
        documents=[
            "doc1",
            "doc2",
            "doc3",
            "doc4",
            "doc5",
            "doc6",
            "doc7",
            "doc8",
        ],
        ids=["id1", "id2", "id3", "id4", "id5", "id6", "id7", "id8"],
    )

    results = cluster.find(query_embeddings=[[1.1, 2.3, 3.2]], n_results=5)

    logger.debug("find result:", results)
    logger.debug(">> create_add_find_em done  !\n")

"""
## Create - Add - Modify - Update
"""
logger.info("## Create - Add - Modify - Update")

def create_add_modify_update(client):
    """
    Create, add, modify, and update

    Parameters
    ----------
    api : _type_
        _description_
    """
    name = "testing"
    new_name = "new_" + name

    cluster = client.get_or_create_cluster(name)

    logger.debug("Before:", cluster.name)
    cluster.modify(name=new_name)
    logger.debug("After:", cluster.name)

    cluster.add(
        documents=[
            "This is document1",
            "This is bidhan",
        ],
        metadatas=[{"source": "notion"}, {"source": "google"}],
        ids=["id1", "id2"],
    )

    logger.debug("Before update:")
    logger.debug(cluster.get(ids=["id1"]))

    cluster.update(ids=["id1"], metadatas=[{"source": "google"}])

    logger.debug("After update source:")
    logger.debug(cluster.get(ids=["id1"]))

    logger.debug(">> create_add_modify_update done !\n")

"""
## Create - Upsert
"""
logger.info("## Create - Upsert")

def create_upsert(client):
    """
    Create and upsert

    Parameters
    ----------
    api : _type_
        _description_
    """
    api.reset()

    name = "testing"

    cluster = client.get_or_create_cluster(name)

    cluster.add(
        documents=[
            "This is document1",
            "This is bidhan",
        ],
        metadatas=[{"source": "notion"}, {"source": "google"}],
        ids=["id1", "id2"],
    )

    cluster.upsert(
        documents=[
            "This is document",
            "This is google",
        ],
        metadatas=[{"source": "notion"}, {"source": "google"}],
        ids=["id1", "id3"],
    )

    logger.debug("Count of documents:", cluster.count())
    logger.debug(">> create_upsert done !\n")

logger.info("\n\n[DONE]", bright=True)