from IPython.display import Markdown, display
from jet.logger import CustomLogger
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/Neo4jVectorDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Neo4j vector store

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Neo4j vector store")

# %pip install llama-index-vector-stores-neo4jvector

# !pip install llama-index


# os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
# openai.api_key = os.environ["OPENAI_API_KEY"]

"""
## Initiate Neo4j vector wrapper
"""
logger.info("## Initiate Neo4j vector wrapper")


username = "neo4j"
password = "pleaseletmein"
url = "bolt://localhost:7687"
embed_dim = 1536

neo4j_vector = Neo4jVectorStore(username, password, url, embed_dim)

"""
## Load documents, build the VectorStoreIndex
"""
logger.info("## Load documents, build the VectorStoreIndex")


"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()


storage_context = StorageContext.from_defaults(vector_store=neo4j_vector)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

query_engine = index.as_query_engine()
response = query_engine.query("What happened at interleaf?")
display(Markdown(f"<b>{response}</b>"))

"""
## Hybrid search

Hybrid search uses a combination of keyword and vector search
In order to use hybrid search, you need to set the `hybrid_search` to `True`
"""
logger.info("## Hybrid search")

neo4j_vector_hybrid = Neo4jVectorStore(
    username, password, url, embed_dim, hybrid_search=True
)

storage_context = StorageContext.from_defaults(
    vector_store=neo4j_vector_hybrid
)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
query_engine = index.as_query_engine()
response = query_engine.query("What happened at interleaf?")
display(Markdown(f"<b>{response}</b>"))

"""
## Load existing vector index

In order to connect to an existing vector index, you need to define the `index_name` and `text_node_property` parameters:

- index_name: name of the existing vector index (default is `vector`)
- text_node_property: name of the property that containt the text value (default is `text`)
"""
logger.info("## Load existing vector index")

index_name = "existing_index"
text_node_property = "text"
existing_vector = Neo4jVectorStore(
    username,
    password,
    url,
    embed_dim,
    index_name=index_name,
    text_node_property=text_node_property,
)

loaded_index = VectorStoreIndex.from_vector_store(existing_vector)

"""
## Customizing responses

You can customize the retrieved information from the knowledge graph using the `retrieval_query` parameter.

The retrieval query must return the following four columns:

* text:str - The text of the returned document
* score:str - similarity score
* id:str - node id
* metadata: Dict - dictionary with additional metadata (must contain `_node_type` and `_node_content` keys)
"""
logger.info("## Customizing responses")

retrieval_query = (
    "RETURN 'Interleaf hired Tomaz' AS text, score, node.id AS id, "
    "{author: 'Tomaz', _node_type:node._node_type, _node_content:node._node_content} AS metadata"
)
neo4j_vector_retrieval = Neo4jVectorStore(
    username, password, url, embed_dim, retrieval_query=retrieval_query
)

loaded_index = VectorStoreIndex.from_vector_store(
    neo4j_vector_retrieval
).as_query_engine()
response = loaded_index.query("What happened at interleaf?")
display(Markdown(f"<b>{response}</b>"))

logger.info("\n\n[DONE]", bright=True)