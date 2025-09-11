from jet.adapters.langchain.chat_ollama import Ollama
from jet.logger import logger
from langchain.chains import GraphQAChain
from langchain_community.graphs import NetworkxEntityGraph
from langchain_community.graphs.index_creator import GraphIndexCreator
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
# NetworkX

>[NetworkX](https://networkx.org/) is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.

This notebook goes over how to do question answering over a graph data structure.

## Setting up

We have to install a Python package.
"""
logger.info("# NetworkX")

# %pip install --upgrade --quiet  networkx

"""
## Create the graph

In this section, we construct an example graph. At the moment, this works best for small pieces of text.
"""
logger.info("## Create the graph")


index_creator = GraphIndexCreator(llm=Ollama(temperature=0))

with open("../../../how_to/state_of_the_union.txt") as f:
    all_text = f.read()

"""
We will use just a small snippet, because extracting the knowledge triplets is a bit intensive at the moment.
"""
logger.info("We will use just a small snippet, because extracting the knowledge triplets is a bit intensive at the moment.")

text = "\n".join(all_text.split("\n\n")[105:108])

text

graph = index_creator.from_text(text)

"""
We can inspect the created graph.
"""
logger.info("We can inspect the created graph.")

graph.get_triples()

"""
## Querying the graph
We can now use the graph QA chain to ask question of the graph
"""
logger.info("## Querying the graph")


chain = GraphQAChain.from_llm(Ollama(temperature=0), graph=graph, verbose=True)

chain.run("what is Intel going to build?")

"""
## Save the graph
We can also save and load the graph.
"""
logger.info("## Save the graph")

graph.write_to_gml("graph.gml")


loaded_graph = NetworkxEntityGraph.from_gml("graph.gml")

loaded_graph.get_triples()

loaded_graph.get_number_of_nodes()

loaded_graph.add_node("NewNode")

loaded_graph.has_node("NewNode")

loaded_graph.remove_node("NewNode")

loaded_graph.get_neighbors("Intel")

loaded_graph.has_edge("Intel", "Silicon Valley")

loaded_graph.remove_edge("Intel", "Silicon Valley")

loaded_graph.clear_edges()

loaded_graph.clear()

logger.info("\n\n[DONE]", bright=True)