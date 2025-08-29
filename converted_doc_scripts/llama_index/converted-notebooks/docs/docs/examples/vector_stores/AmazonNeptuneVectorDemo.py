from IPython.display import Markdown, display
from jet.logger import CustomLogger
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.neptune import NeptuneAnalyticsVectorStore
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Amazon Neptune - Neptune Analytics vector store

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Amazon Neptune - Neptune Analytics vector store")

# %pip install llama-index-vector-stores-neptune

"""
## Initiate Neptune Analytics vector wrapper
"""
logger.info("## Initiate Neptune Analytics vector wrapper")


graph_identifier = ""
embed_dim = 1536

neptune_vector_store = NeptuneAnalyticsVectorStore(
    graph_identifier=graph_identifier, embedding_dimension=1536
)

"""
## Load documents, build the VectorStoreIndex
"""
logger.info("## Load documents, build the VectorStoreIndex")


"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()


storage_context = StorageContext.from_defaults(
    vector_store=neptune_vector_store
)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

query_engine = index.as_query_engine()
response = query_engine.query("What happened at interleaf?")
display(Markdown(f"<b>{response}</b>"))

logger.info("\n\n[DONE]", bright=True)