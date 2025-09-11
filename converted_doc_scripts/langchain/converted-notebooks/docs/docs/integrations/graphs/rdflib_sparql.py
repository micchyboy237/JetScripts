from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.chains import GraphSparqlQAChain
from langchain_community.graphs import RdfGraph
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
# RDFLib

>[RDFLib](https://rdflib.readthedocs.io/) is a pure Python package for working with [RDF](https://en.wikipedia.org/wiki/Resource_Description_Framework). `RDFLib` contains most things you need to work with `RDF`, including:
>- parsers and serializers for RDF/XML, N3, NTriples, N-Quads, Turtle, TriX, Trig and JSON-LD
>- a Graph interface which can be backed by any one of a number of Store implementations
>- store implementations for in-memory, persistent on disk (Berkeley DB) and remote SPARQL endpoints
>- a SPARQL 1.1 implementation - supporting SPARQL 1.1 Queries and Update statements
>- SPARQL function extension mechanisms

Graph databases are an excellent choice for applications based on network-like models. To standardize the syntax and semantics of such graphs, the W3C recommends `Semantic Web Technologies`, cp. [Semantic Web](https://www.w3.org/standards/semanticweb/). 

[SPARQL](https://www.w3.org/TR/sparql11-query/) serves as a query language analogously to `SQL` or `Cypher` for these graphs. This notebook demonstrates the application of LLMs as a natural language interface to a graph database by generating `SPARQL`.

**Disclaimer:** To date, `SPARQL` query generation via LLMs is still a bit unstable. Be especially careful with `UPDATE` queries, which alter the graph.

## Setting up

We have to install a python library:
"""
logger.info("# RDFLib")

# !pip install rdflib

"""
There are several sources you can run queries against, including files on the web, files you have available locally, SPARQL endpoints, e.g., [Wikidata](https://www.wikidata.org/wiki/Wikidata:Main_Page), and [triple stores](https://www.w3.org/wiki/LargeTripleStores).
"""
logger.info("There are several sources you can run queries against, including files on the web, files you have available locally, SPARQL endpoints, e.g., [Wikidata](https://www.wikidata.org/wiki/Wikidata:Main_Page), and [triple stores](https://www.w3.org/wiki/LargeTripleStores).")


graph = RdfGraph(
    source_file="http://www.w3.org/People/Berners-Lee/card",
    standard="rdf",
    local_copy="test.ttl",
)

"""
Note that providing a `local_file` is necessary for storing changes locally if the source is read-only.

## Refresh graph schema information
If the schema of the database changes, you can refresh the schema information needed to generate SPARQL queries.
"""
logger.info("## Refresh graph schema information")

graph.load_schema()

graph.get_schema

"""
## Querying the graph

Now, you can use the graph SPARQL QA chain to ask questions about the graph.
"""
logger.info("## Querying the graph")

chain = GraphSparqlQAChain.from_llm(
    ChatOllama(model="llama3.2"), graph=graph, verbose=True
)

chain.run("What is Tim Berners-Lee's work homepage?")

"""
## Updating the graph

Analogously, you can update the graph, i.e., insert triples, using natural language.
"""
logger.info("## Updating the graph")

chain.run(
    "Save that the person with the name 'Timothy Berners-Lee' has a work homepage at 'http://www.w3.org/foo/bar/'"
)

"""
Let's verify the results:
"""
logger.info("Let's verify the results:")

query = (
    """PREFIX foaf: <http://xmlns.com/foaf/0.1/>\n"""
    """SELECT ?hp\n"""
    """WHERE {\n"""
    """    ?person foaf:name "Timothy Berners-Lee" . \n"""
    """    ?person foaf:workplaceHomepage ?hp .\n"""
    """}"""
)
graph.query(query)

"""
## Return SPARQL query
You can return the SPARQL query step from the Sparql QA Chain using the `return_sparql_query` parameter
"""
logger.info("## Return SPARQL query")

chain = GraphSparqlQAChain.from_llm(
    ChatOllama(model="llama3.2"), graph=graph, verbose=True, return_sparql_query=True
)

result = chain("What is Tim Berners-Lee's work homepage?")
logger.debug(f"SPARQL query: {result['sparql_query']}")
logger.debug(f"Final answer: {result['result']}")

logger.debug(result["sparql_query"])

logger.info("\n\n[DONE]", bright=True)