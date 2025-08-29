from jet.models.config import MODELS_CACHE_DIR
from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.objectbox import ObjectBoxVectorStore
from objectbox import VectorDistanceType
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# ObjectBox VectorStore Demo

This notebook will demonstrate the use of [ObjectBox](https://objectbox.io/) as an efficient, on-device vector-store with LlamaIndex. We will consider a simple RAG use-case where given a document, the user can ask questions and get relevant answers from a LLM in natural language. The RAG pipeline will be configured along the following verticals:

* A builtin [`SimpleDirectoryReader` reader](https://docs.llamaindex.ai/en/stable/examples/data_connectors/simple_directory_reader/) from LlamaIndex
* A builtin [`SentenceSplitter` node-parser](https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/sentence_splitter/) from LlamaIndex
* Models from [HuggingFace as embedding providers](https://docs.llamaindex.ai/en/stable/examples/embeddings/huggingface/)
* [ObjectBox](https://objectbox.io/) as NoSQL and vector datastore
* Google's [Gemini](https://docs.llamaindex.ai/en/stable/examples/llm/gemini/) as a remote LLM service

## 1) Installing dependencies

We install integrations for HuggingFace and Gemini to use along with LlamaIndex
"""
logger.info("# ObjectBox VectorStore Demo")

# !pip install llama_index_vector_stores_objectbox --quiet
# !pip install llama-index --quiet
# !pip install llama-index-embeddings-huggingface --quiet
# !pip install llama-index-llms-gemini --quiet

"""
## 2) Downloading the documents
"""
logger.info("## 2) Downloading the documents")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
## 3) Setup a LLM for RAG (Gemini)

We use Google Gemini's cloud-based API as a LLM. You can get an API-key from the [console](https://aistudio.google.com/app/apikey).
"""
logger.info("## 3) Setup a LLM for RAG (Gemini)")

# import getpass

# gemini_key_api = getpass.getpass("Gemini API Key: ")
gemini_llm = Gemini(api_key=gemini_key_api)

"""
## 4) Setup an embedding model for RAG (HuggingFace `bge-small-en-v1.5`)

HuggingFace hosts a variety of embedding models, which could be observed from the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard).
"""
logger.info("## 4) Setup an embedding model for RAG (HuggingFace `bge-small-en-v1.5`)")


hf_embedding = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)
embedding_dim = 384

"""
## 5) Prepare documents and nodes

In a RAG pipeline, the first step is to read the given documents. We use the `SimpleDirectoryReader` that selects the best file reader by checking the file extension from the directory.

Next, we produce chunks (text subsequences) from the contents read by the `SimpleDirectoryReader` from the documents. A `SentenceSplitter` is a text-splitter that preserves sentence boundaries while splitting the text into chunks of size `chunk_size`.
"""
logger.info("## 5) Prepare documents and nodes")


reader = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data")
documents = reader.load_data()

node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
nodes = node_parser.get_nodes_from_documents(documents)

"""
## 6) Configure `ObjectBoxVectorStore`

The `ObjectBoxVectorStore` can be initialized with several options:

- `embedding_dim` (required): The dimensions of the embeddings that the vector DB will hold
- `distance_type`: Choose from `COSINE`, `DOT_PRODUCT`, `DOT_PRODUCT_NON_NORMALIZED` and `EUCLIDEAN`
- `db_directory`: The path of the directory where the `.mdb` ObjectBox database file should be created
- `clear_db`: Deletes the existing database file if it exists on `db_directory`
- `do_log`: Enables logging from the ObjectBox integration
"""
logger.info("## 6) Configure `ObjectBoxVectorStore`")


vector_store = ObjectBoxVectorStore(
    embedding_dim,
    distance_type=VectorDistanceType.COSINE,
    db_directory="obx_data",
    clear_db=False,
    do_log=True,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

Settings.llm = gemini_llm
Settings.embed_model = hf_embedding

index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)

"""
## 7) Chat with the document
"""
logger.info("## 7) Chat with the document")

query_engine = index.as_query_engine()
response = query_engine.query("Who is Paul Graham?")
logger.debug(response)

"""
## Optional: Configuring `ObjectBoxVectorStore` as a retriever

A LlamaIndex [retriever](https://docs.llamaindex.ai/en/stable/module_guides/querying/retriever/) is responsible for fetching similar chunks from a vector DB given a query.
"""
logger.info("## Optional: Configuring `ObjectBoxVectorStore` as a retriever")

retriever = index.as_retriever()
response = retriever.retrieve("What did the author do growing up?")

for node in response:
    logger.debug("Retrieved chunk text:\n", node.node.get_text())
    logger.debug("Retrieved chunk metadata:\n", node.node.get_metadata_str())
    logger.debug("\n\n\n")

"""
## Optional: Removing chunks associated with a single query using `delete_nodes`

We can use the `ObjectBoxVectorStore.delete_nodes` method to remove chunks (nodes) from the vector DB providing a list containing node IDs as an argument.
"""
logger.info("## Optional: Removing chunks associated with a single query using `delete_nodes`")

response = retriever.retrieve("What did the author do growing up?")

node_ids = []
for node in response:
    node_ids.append(node.node_id)
logger.debug(f"Nodes to be removed: {node_ids}")

logger.debug(f"No. of vectors before deletion: {vector_store.count()}")
vector_store.delete_nodes(node_ids)
logger.debug(f"No. of vectors after deletion: {vector_store.count()}")

"""
## Optional: Removing a single document from the vector DB

The `ObjectBoxVectorStore.delete` method can be used to remove chunks (nodes) associated with a single document whose `id_` is provided as an argument.
"""
logger.info("## Optional: Removing a single document from the vector DB")

document = documents[0]
logger.debug(f"Document to be deleted {document.id_}")

logger.debug(f"No. of vectors before deletion: {vector_store.count()}")
vector_store.delete(document.id_)
logger.debug(f"No. of vectors after document deletion: {vector_store.count()}")

logger.info("\n\n[DONE]", bright=True)