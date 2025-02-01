from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
from langchain_community.graphs.index_creator import GraphIndexCreator
from jet.llm.ollama.base_langchain import Ollama
# from langchain.chains import GraphQAChain
from langchain.chains.graph_qa.base import GraphQAChain
from langchain_community.graphs import NetworkxEntityGraph

initialize_ollama_settings()

"""
# NetworkX

>[NetworkX](https://networkx.org/) is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.

This notebook goes over how to do question answering over a graph data structure.

## Setting up

We have to install a Python package.
"""

# %pip install --upgrade --quiet  networkx

"""
## Create the graph

In this section, we construct an example graph. At the moment, this works best for small pieces of text.
"""


index_creator = GraphIndexCreator(llm=Ollama(temperature=0))

data_file = "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/langchain/docs/docs/how_to/state_of_the_union.txt"

with open(data_file) as f:
    all_text = f.read()

"""
We will use just a small snippet, because extracting the knowledge triplets is a bit intensive at the moment.
"""

text = "\n".join(all_text.split("\n\n"))

logger.newline()
logger.debug("Result 1:")
logger.success(text)

graph = index_creator.from_text(text)

"""
We can inspect the created graph.
"""

logger.newline()
logger.debug("Result 2:")
logger.success(graph.get_triples())

"""
## Querying the graph
We can now use the graph QA chain to ask question of the graph
"""


chain = GraphQAChain.from_llm(Ollama(temperature=0), graph=graph, verbose=True)

result = chain.run("what is Intel going to build?")

logger.newline()
logger.debug("Result 3:")
logger.success(result)

"""
## Save the graph
We can also save and load the graph.
"""

logger.newline()
graph.write_to_gml("graph.gml")


loaded_graph = NetworkxEntityGraph.from_gml("graph.gml")

logger.newline()
logger.debug("Result 4:")
logger.success(loaded_graph.get_triples())

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
