"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/query_engine/knowledge_graph_query_engine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

"""
# Knowledge Graph Query Engine

Creating a Knowledge Graph usually involves specialized and complex tasks. However, by utilizing the Llama Index (LLM), the KnowledgeGraphIndex, and the GraphStore, we can facilitate the creation of a relatively effective Knowledge Graph from any data source supported by [Llama Hub](https://llamahub.ai/).

Furthermore, querying a Knowledge Graph often requires domain-specific knowledge related to the storage system, such as Cypher. But, with the assistance of the LLM and the LlamaIndex KnowledgeGraphQueryEngine, this can be accomplished using Natural Language!

In this demonstration, we will guide you through the steps to:

- Extract and Set Up a Knowledge Graph using the Llama Index
- Query a Knowledge Graph using Cypher
- Query a Knowledge Graph using Natural Language
"""

"""
If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""

# %pip install llama-index-readers-wikipedia
# %pip install llama-index-llms-azure-openai
# %pip install llama-index-graph-stores-nebula
# %pip install llama-index-llms-ollama
# %pip install llama-index-embeddings-azure-openai

# !pip install llama-index

"""
Let's first get ready for basic preparation of Llama Index.
"""

"""
### Ollama
"""


# os.environ["OPENAI_API_KEY"] = "sk-..."


from llama_index.core.query_engine import KnowledgeGraphQueryEngine
from llama_index.core import KnowledgeGraphIndex
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core import download_loader
from llama_index.graph_stores.nebula import NebulaGraphStore
from llama_index.core import StorageContext
from llama_index.embeddings.azure_openai import AzureOllamaEmbedding
from llama_index.llms.azure_openai import AzureOllama
from llama_index.core import Settings
from jet.llm.ollama.base import Ollama
import os
import logging
import sys
logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output


Settings.llm = Ollama(temperature=0, model="llama3.2",
                      request_timeout=300.0, context_window=4096)
Settings.chunk_size = 512

"""
### Azure
"""


api_key = "<api-key>"
azure_endpoint = "https://<your-resource-name>.openai.azure.com/"
api_version = "2023-07-01-preview"

llm = AzureOllama(
    model="gpt-35-turbo-16k",
    deployment_name="my-custom-llm",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

embed_model = AzureOllamaEmbedding(
    model="text-embedding-ada-002",
    deployment_name="my-custom-embedding",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)


Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512

"""
## Prepare for NebulaGraph

Before next step to creating the Knowledge Graph, let's ensure we have a running NebulaGraph with defined data schema.
"""

# %pip install ipython-ngql nebula3-python

os.environ["NEBULA_USER"] = "root"
os.environ["NEBULA_PASSWORD"] = "nebula"  # default is "nebula"
os.environ[
    "NEBULA_ADDRESS"
] = "127.0.0.1:9669"  # assumed we have NebulaGraph installed locally

space_name = "llamaindex"
edge_types, rel_prop_names = ["relationship"], [
    "relationship"
]  # default, could be omit if create from an empty kg
tags = ["entity"]  # default, could be omit if create from an empty kg

"""
Prepare for StorageContext with graph_store as NebulaGraphStore
"""


graph_store = NebulaGraphStore(
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
)
storage_context = StorageContext.from_defaults(graph_store=graph_store)

"""
## (Optional)Build the Knowledge Graph with LlamaIndex

With the help of Llama Index and LLM defined, we could build Knowledge Graph from given documents.

If we have a Knowledge Graph on NebulaGraphStore already, this step could be skipped
"""

"""
### Step 1, load data from Wikipedia for "Guardians of the Galaxy Vol. 3"
"""


loader = WikipediaReader()

documents = loader.load_data(
    pages=["Guardians of the Galaxy Vol. 3"], auto_suggest=False
)

"""
### Step 2, Generate a KnowledgeGraphIndex with NebulaGraph as graph_store

Then, we will create a KnowledgeGraphIndex to enable Graph based RAG, see [here](https://gpt-index.readthedocs.io/en/latest/examples/index_structs/knowledge_graph/KnowledgeGraphIndex_vs_VectorStoreIndex_vs_CustomIndex_combined.html) for deails, apart from that, we have a Knowledge Graph up and running for other purposes, too!
"""


kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=10,
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
    include_embeddings=True,
)

"""
Now we have a Knowledge Graph on NebulaGraph cluster under space named `llamaindex` about the 'Guardians of the Galaxy Vol. 3' movie, let's play with it a little bit.
"""

# %pip install ipython-ngql networkx pyvis
# %load_ext ngql
# %ngql --address 127.0.0.1 --port 9669 --user root --password <password>

# %ngql USE llamaindex;
# %ngql MATCH ()-[e]->() RETURN e LIMIT 10

# %ng_draw

"""
## Asking the Knowledge Graph

Finally, let's demo how to Query Knowledge Graph with Natural language!

Here, we will leverage the `KnowledgeGraphQueryEngine`, with `NebulaGraphStore` as the `storage_context.graph_store`.
"""


query_engine = KnowledgeGraphQueryEngine(
    storage_context=storage_context,
    llm=llm,
    verbose=True,
)

response = query_engine.query(
    "Tell me about Peter Quill?",
)
display(Markdown(f"<b>{response}</b>"))

graph_query = query_engine.generate_query(
    "Tell me about Peter Quill?",
)

graph_query = graph_query.replace("WHERE", "\n  WHERE").replace(
    "RETURN", "\nRETURN"
)

display(
    Markdown(
        f"""
```cypher
{graph_query}
```
"""
    )
)

"""
We could see it helps generate the Graph query:

```cypher
MATCH (p:`entity`)-[:relationship]->(e:`entity`) 
  WHERE p.`entity`.`name` == 'Peter Quill' 
RETURN e.`entity`.`name`;
```
And synthese the question based on its result:

```json
{'e2.entity.name': ['grandfather', 'alternate version of Gamora', 'Guardians of the Galaxy']}
```
"""

"""
Of course we still could query it, too! And this query engine could be our best Graph Query Language learning bot, then :).
"""

# %%ngql
MATCH(p: `entity`)-[e:relationship] -> (m: `entity`)
WHERE p.`entity`.`name` == 'Peter Quill'
RETURN p.`entity`.`name`, e.relationship, m.`entity`.`name`

"""
And change the query to be rendered
"""

# %%ngql
MATCH(p: `entity`)-[e:relationship] -> (m: `entity`)
WHERE p.`entity`.`name` == 'Peter Quill'
RETURN p, e, m

# %ng_draw

"""
The results of this knowledge-fetching query could not be more clear from the renderred graph then.
"""
