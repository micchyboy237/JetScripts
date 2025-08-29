from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core import ComposableGraph
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core import SummaryIndex
from llama_index.core import VectorStoreIndex, SimpleKeywordTableIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
import logging
import os
import shutil
import sys


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Docstore Demo

This guide shows you how to directly use our `DocumentStore` abstraction. By putting nodes in the docstore, this allows you to define multiple indices over the same underlying docstore, instead of duplicating data across indices.

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/docstore/DocstoreDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Docstore Demo")

# %pip install llama-index-llms-ollama

# !pip install llama-index

# import nest_asyncio

# nest_asyncio.apply()


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


"""
#### Download Data
"""
logger.info("#### Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
#### Load Documents
"""
logger.info("#### Load Documents")

reader = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/")
documents = reader.load_data()

"""
#### Parse into Nodes
"""
logger.info("#### Parse into Nodes")


nodes = SentenceSplitter().get_nodes_from_documents(documents)

"""
#### Add to Docstore
"""
logger.info("#### Add to Docstore")


docstore = SimpleDocumentStore()
docstore.add_documents(nodes)

"""
#### Define Multiple Indexes

Each index uses the same underlying Node.
"""
logger.info("#### Define Multiple Indexes")



storage_context = StorageContext.from_defaults(docstore=docstore)
summary_index = SummaryIndex(nodes, storage_context=storage_context)
vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
keyword_table_index = SimpleKeywordTableIndex(
    nodes, storage_context=storage_context
)

len(storage_context.docstore.docs)

"""
#### Test out some Queries
"""
logger.info("#### Test out some Queries")

llm = OllamaFunctionCallingAdapter(temperature=0, model="llama3.2", request_timeout=300.0, context_window=4096)

Settings.llm = llm
Settings.chunk_size = 1024

query_engine = summary_index.as_query_engine()
response = query_engine.query("What is a summary of this document?")

query_engine = vector_index.as_query_engine()
response = query_engine.query("What did the author do growing up?")

query_engine = keyword_table_index.as_query_engine()
response = query_engine.query("What did the author do after his time at YC?")

logger.debug(response)

logger.info("\n\n[DONE]", bright=True)