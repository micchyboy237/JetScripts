from jet.logger import CustomLogger
from llama_index import (
ServiceContext,
VectorStoreIndex,
SummaryIndex,
)
from pprint import pprint
import llama_hub.docstring_walker as docstring_walker
import os
import shutil
import torch_geometric.nn.kge as kge


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-docstring-walker/examples/docstringwalker_example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Intro

This notebook will show you an example of how to use DocstringWalker from Llama Hub, combined with Llama Index and LLM of your choice.

# Lib install for Collab
"""
logger.info("# Intro")

# !pip install llama_index

# !pip install llama_hub

"""
For this exercise we will use **PyTorch Geometric (PyG)** module for inspecting multi-module doctstrings.
"""
logger.info("For this exercise we will use **PyTorch Geometric (PyG)** module for inspecting multi-module doctstrings.")

# !pip install torch_geometric

"""
# Lib imports
"""
logger.info("# Lib imports")





"""
# Example 1 - reading Docstring Walker's own docstrings

Let's start by using it.... on itself :) We will see what information gets extracted from the module.
"""
logger.info("# Example 1 - reading Docstring Walker's own docstrings")

walker = docstring_walker.DocstringWalker()

path_to_docstring_walker = os.path.dirname(docstring_walker.__file__)

example1_docs = walker.load_data(path_to_docstring_walker)

logger.debug(example1_docs[0].text)

"""
Now we can use the doc to generate Llama index and use it with LLM.
"""
logger.info("Now we can use the doc to generate Llama index and use it with LLM.")

example1_index = VectorStoreIndex(example1_docs)

example1_query_engine = example1_index.as_query_engine()

plogger.debug(
    example1_query_engine.query("What is the main purpose of DocstringWalker?").response
)

logger.debug(
    example1_query_engine.query(
        "What are the main functions used in DocstringWalker. Use numbered list, briefly describe each function."
    ).response
)

"""
# Example 2 - checking multi-module project

Now we can use the same approach to check a multi-module project. Let's use **PyTorch Geometric (PyG) Knowledge Graph (KG)** module for this exercise.
"""
logger.info("# Example 2 - checking multi-module project")


path_to_module = os.path.dirname(kge.__file__)
example2_docs = walker.load_data(path_to_module)

example2_index = SummaryIndex(example2_docs)
example2_docs = example2_index.as_query_engine()

logger.debug(
    example2_docs.query(
        "What classes are available and what is their main purpose? Use nested numbered list to describe: the class name, short summary of purpose, papers or literature review for each one of them"
    ).response
)

logger.debug(example2_docs.query("What are the parameters required by TransE class?").response)

logger.info("\n\n[DONE]", bright=True)