from typing import Literal
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core import PropertyGraphIndex
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.graph_stores.kuzu import KuzuPropertyGraphStore
import kuzu
import shutil
from llama_index.core import SimpleDirectoryReader
import os
import nest_asyncio
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()


# [Kùzu](https://kuzudb.com/) is an open source, embedded graph database that's designed for query speed and scalability. It implements the Cypher query language, and utilizes a structured property graph model (a variant of the labelled property graph model) with support for ACID transactions. Because Kùzu is embedded, there's no requirement for a server to set up and use the database.
#
# If you already have an existing graph, please skip to the end of this notebook. Otherwise, let's begin by creating a graph from unstructured text to demonstrate how to use Kùzu as a graph store.


nest_asyncio.apply()

# Environment Setup


# os.environ["OPENAI_API_KEY"] = "enter your key here"

# We will be using Ollama models for this example, so we'll specify the Ollama API key.

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'


documents = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/summaries/").load_data()

# Graph Construction
#
# We first need to create an empty Kùzu database directory by calling the `kuzu.Database` constructor. This step instantiates the database and creates the necessary directories and files within a local directory that stores the graph. This `Database` object is then passed to the `KuzuPropertyGraph` constructor.


shutil.rmtree("test_db", ignore_errors=True)
db = kuzu.Database("test_db")


graph_store = KuzuPropertyGraphStore(db)

# Because Kùzu implements the structured graph property model, it imposes some level of structure on the schema of the graph. In the above case, because we did not specify a relationship schema that we want in our graph, it uses a generic schema, where the relationship types are not constrained, allowing the extracted triples from the LLM to be stored as relationships in the graph.

# Define models
#
# Below, we'll define the models used for embedding the text and the LLMs that are used to extract triples from the text and generate the response.
# In this case, we specify different temperature settings for the same model - the extraction model has a temperature of 0.


embed_model = OllamaEmbedding(model_name="nomic-embed-text")
extract_llm = Ollama(model="llama3.1", request_timeout=300.0,
                     context_window=4096, temperature=0.0)
generate_llm = Ollama(model="llama3.1", request_timeout=300.0,
                      context_window=4096, temperature=0.3)

# 1. Create property graph index without imposing structure
#
# Because we didn't specify the relationship schema above, we can simply invoke the `SchemaLLMPathExtractor` to extract the triples from the text and store them in the graph. We can define the property graph index using Kùzu as the graph store, as shown below:


index = PropertyGraphIndex.from_documents(
    documents,
    embed_model=embed_model,
    kg_extractors=[SchemaLLMPathExtractor(extract_llm)],
    property_graph_store=graph_store,
    show_progress=True,
)

# Now that the graph is created, we can explore it in [Kùzu Explorer](https://docs.kuzudb.com/visualization/), a web-base UI, by running a Docker container that pulls the latest image of Kùzu Explorer as follows:
# ```bash
# docker run -p 8000:8000 \
#            -v ./test_db:/database \
#            --rm kuzudb/explorer:latest
# ```
#
# Then, launch the UI and then visting [http://localhost:8000/](http://localhost:8000/).
#
# The easiest way to see the entire graph is to use a Cypher query like `"match (a)-[b]->(c) return * limit 200"`.
#
# To delete the entire graph, you can either delete the `./test_db` directory that contains the database files, or run the Cypher query `"match (n) detach delete n"` in the Kùzu Explorer shell.

# Querying and Retrieval

Settings.llm = generate_llm

query_engine = index.as_query_engine(include_text=False)

response = query_engine.query("Tell me more about Interleaf and Viaweb")

print(str(response))

# 2. Create property graph index with structure
#
# The recommended way to use Kùzu is to apply a structured schema to the graph. The schema is defined by specifying the relationship types (including direction) that we want in the graph. The imposition of structure helps with generating triples that are more meaningful for the types of questions we may want to answer from the graph.
#
# By specifying the below validation schema, we can enforce that the graph only contains relationships of the specified types.


entities = Literal["PERSON", "PLACE", "ORGANIZATION"]
relations = Literal["HAS", "PART_OF", "WORKED_ON", "WORKED_WITH", "WORKED_AT"]
validation_schema = [
    ("ORGANIZATION", "HAS", "PERSON"),
    ("PERSON", "WORKED_AT", "ORGANIZATION"),
    ("PERSON", "WORKED_WITH", "PERSON"),
    ("PERSON", "WORKED_ON", "ORGANIZATION"),
    ("PERSON", "PART_OF", "ORGANIZATION"),
    ("ORGANIZATION", "PART_OF", "ORGANIZATION"),
    ("PERSON", "WORKED_AT", "PLACE"),
]

shutil.rmtree("test_db", ignore_errors=True)
db = kuzu.Database("test_db")

# Along with the `Database` constructor, we also specify two additional arguments to the property graph store: `has_structured_schema=True` and `relationship_schema=validation_schema`, which provides Kùzu additional information as it instantiates a new graph.

graph_store = KuzuPropertyGraphStore(
    db,
    has_structured_schema=True,
    relationship_schema=validation_schema,
)

# To construct a property graph with the desired schema, observe that we specify a few additional arguments to the `SchemaLLMPathExtractor`.

index = PropertyGraphIndex.from_documents(
    documents,
    embed_model=OllamaEmbedding(model_name="nomic-embed-text"),
    kg_extractors=[
        SchemaLLMPathExtractor(
            llm=Ollama(model="llama3.1", request_timeout=300.0,
                       context_window=4096, temperature=0.0),
            possible_entities=entities,
            possible_relations=relations,
            kg_validation_schema=validation_schema,
            strict=True,  # if false, will allow triples outside of the schema
        )
    ],
    property_graph_store=graph_store,
    show_progress=True,
)

# We can now apply the query engine on the index as before.

Settings.llm = generate_llm

query_engine = index.as_query_engine(include_text=False)

response2 = query_engine.query("Tell me more about Interleaf and Viaweb")
print(str(response2))

# Use existing graph
#
# You can reuse an existing `Database` object to connect to its underlying `PropertyGraphIndex`. This is useful when you want to query the graph without having to re-extract the triples from the text.

graph_store = KuzuPropertyGraphStore(db)

index = PropertyGraphIndex.from_existing(
    embed_model=embed_model,
    llm=generate_llm,
    property_graph_store=graph_store,
)

query_engine = index.as_query_engine(include_text=False)

response3 = query_engine.query("When was Viaweb founded, and by whom?")
print(str(response3))

# For full details on construction, retrieval, querying of a property graph, see the [full docs page](/../../module_guides/indexing/lpg_index_guide).

logger.info("\n\n[DONE]", bright=True)
