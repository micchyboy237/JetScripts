from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import PropertyGraphIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.memgraph import MemgraphPropertyGraphStore
import os
import shutil
import urllib.request


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# Memgraph Property Graph Index

[Memgraph](https://memgraph.com/) is an open source graph database built real-time streaming and fast analysis of your stored data.

Before running Memgraph, ensure you have Docker running in the background. The quickest way to try out [Memgraph Platform](https://memgraph.com/docs/getting-started#install-memgraph-platform) (Memgraph database + MAGE library + Memgraph Lab) for the first time is running the following command:

For Linux/macOS:
```shell
curl https://install.memgraph.com | sh
```

For Windows:
```shell
iwr https://windows.memgraph.com | iex
```

From here, you can check Memgraph's visual tool, Memgraph Lab on the [http://localhost:3000/](http://localhost:3000/) or the [desktop version](https://memgraph.com/download) of the app.
"""
logger.info("# Memgraph Property Graph Index")

# %pip install llama-index llama-index-graph-stores-memgraph

"""
## Environment setup
"""
logger.info("## Environment setup")


os.environ[
#     "OPENAI_API_KEY"
] = "sk-proj-..."  # Replace with your MLX API key

"""
Create the data directory and download the Paul Graham essay we'll be using as the input data for this example.
"""
logger.info("Create the data directory and download the Paul Graham essay we'll be using as the input data for this example.")


os.makedirs(f"{GENERATED_DIR}/paul_graham/", exist_ok=True)

url = "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt"
output_path = f"{GENERATED_DIR}/paul_graham/paul_graham_essay.txt"
urllib.request.urlretrieve(url, output_path)

# import nest_asyncio

# nest_asyncio.apply()

"""
Read the file, replace single quotes, save the modified content and load the document data using the `SimpleDirectoryReader`
"""
logger.info("Read the file, replace single quotes, save the modified content and load the document data using the `SimpleDirectoryReader`")


with open(output_path, "r", encoding="utf-8") as file:
    content = file.read()

with open(output_path, "w", encoding="utf-8") as file:
    file.write(content)

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

"""
## Setup Memgraph connection

Set up your graph store class by providing the database credentials.
"""
logger.info("## Setup Memgraph connection")


username = ""  # Enter your Memgraph username (default "")
password = ""  # Enter your Memgraph password (default "")
url = ""  # Specify the connection URL, e.g., 'bolt://localhost:7687'

graph_store = MemgraphPropertyGraphStore(
    username=username,
    password=password,
    url=url,
)

"""
## Index Construction
"""
logger.info("## Index Construction")


index = PropertyGraphIndex.from_documents(
    documents,
    embed_model=MLXEmbedding(model_name="text-embedding-ada-002"),
    kg_extractors=[
        SchemaLLMPathExtractor(
            llm=MLXLlamaIndexLLMAdapter(model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats", temperature=0.0)
        )
    ],
    property_graph_store=graph_store,
    show_progress=True,
)

"""
Now that the graph is created, we can explore it in the UI by visiting [http://localhost:3000/](http://localhost:3000/).

The easiest way to visualize the entire graph is by running a Cypher command similar to this:

```shell
MATCH p=()-[]-() RETURN p;
```

This command matches all of the possible paths in the graph and returns entire graph. 

To visualize the schema of the graph, visit the Graph schema tab and generate the new schema based on the newly created graph.

To delete an entire graph, use:

```shell
MATCH (n) DETACH DELETE n;
```

## Querying and retrieval
"""
logger.info("## Querying and retrieval")

retriever = index.as_retriever(include_text=False)

nodes = retriever.retrieve("What happened at Interleaf and Viaweb?")

logger.debug("Query Results:")
for node in nodes:
    logger.debug(node.text)

query_engine = index.as_query_engine(include_text=True)

response = query_engine.query("What happened at Interleaf and Viaweb?")
logger.debug("\nDetailed Query Response:")
logger.debug(str(response))

"""
## Loading from an existing graph

If you have an existing graph (either created with LlamaIndex or otherwise), we can connect to and use it!

**NOTE:** If your graph was created outside of LlamaIndex, the most useful retrievers will be [text to cypher](/../../module_guides/indexing/lpg_index_guide#texttocypherretriever) or [cypher templates](/../../module_guides/indexing/lpg_index_guide#cyphertemplateretriever). Other retrievers rely on properties that LlamaIndex inserts.
"""
logger.info("## Loading from an existing graph")

llm = MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats", temperature=0.0)
kg_extractors = [SchemaLLMPathExtractor(llm=llm)]

index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store,
    kg_extractors=kg_extractors,
    embed_model=MLXEmbedding(model_name="text-embedding-ada-002"),
    show_progress=True,
)

logger.info("\n\n[DONE]", bright=True)