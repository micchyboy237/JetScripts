from IPython.display import Markdown, display
from jet.logger import CustomLogger
from llama_index.core import (
VectorStoreIndex,
SimpleDirectoryReader,
load_index_from_storage,
StorageContext,
)
from llama_index.core import Document
from llama_index.core import QueryBundle
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
import llama_index.core
import logging
import nltk
import openai
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/vector_stores/SimpleIndexDemo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Simple Vector Store

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Simple Vector Store")

# !pip install llama-index


# os.environ["OPENAI_API_KEY"] = "sk-..."
# openai.api_key = os.environ["OPENAI_API_KEY"]

"""
#### Load documents, build the VectorStoreIndex
"""
logger.info("#### Load documents, build the VectorStoreIndex")


nltk.download("stopwords")



logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()

index = VectorStoreIndex.from_documents(documents)

index.set_index_id("vector_index")
index.storage_context.persist("./storage")

storage_context = StorageContext.from_defaults(persist_dir="storage")
index = load_index_from_storage(storage_context, index_id="vector_index")

"""
#### Query Index
"""
logger.info("#### Query Index")

query_engine = index.as_query_engine(response_mode="tree_summarize")
response = query_engine.query("What did the author do growing up?")

display(Markdown(f"<b>{response}</b>"))

"""
**Query Index with SVM/Linear Regression**

Use Karpathy's [SVM-based](https://twitter.com/karpathy/status/1647025230546886658?s=20) approach. Set query as positive example, all other datapoints as negative examples, and then fit a hyperplane.
"""
logger.info("Use Karpathy's [SVM-based](https://twitter.com/karpathy/status/1647025230546886658?s=20) approach. Set query as positive example, all other datapoints as negative examples, and then fit a hyperplane.")

query_modes = [
    "svm",
    "linear_regression",
    "logistic_regression",
]
for query_mode in query_modes:
    query_engine = index.as_query_engine(vector_store_query_mode=query_mode)
    response = query_engine.query("What did the author do growing up?")
    logger.debug(f"Query mode: {query_mode}")
    display(Markdown(f"<b>{response}</b>"))

display(Markdown(f"<b>{response}</b>"))

logger.debug(response.source_nodes[0].text)

"""
**Query Index with custom embedding string**
"""


query_bundle = QueryBundle(
    query_str="What did the author do growing up?",
    custom_embedding_strs=["The author grew up painting."],
)
query_engine = index.as_query_engine()
response = query_engine.query(query_bundle)

display(Markdown(f"<b>{response}</b>"))

"""
**Use maximum marginal relevance**

Instead of ranking vectors purely by similarity, adds diversity to the documents by penalizing documents similar to ones that have already been found based on <a href="https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf">MMR</a> . A lower mmr_treshold increases diversity.
"""
logger.info("Instead of ranking vectors purely by similarity, adds diversity to the documents by penalizing documents similar to ones that have already been found based on <a href="https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf">MMR</a> . A lower mmr_treshold increases diversity.")

query_engine = index.as_query_engine(
    vector_store_query_mode="mmr", vector_store_kwargs={"mmr_threshold": 0.2}
)
response = query_engine.query("What did the author do growing up?")

"""
#### Get Sources
"""
logger.info("#### Get Sources")

logger.debug(response.get_formatted_sources())

"""
#### Query Index with Filters

We can also filter our queries using metadata
"""
logger.info("#### Query Index with Filters")


doc = Document(text="target", metadata={"tag": "target"})

index.insert(doc)


filters = MetadataFilters(
    filters=[ExactMatchFilter(key="tag", value="target")]
)

retriever = index.as_retriever(
    similarity_top_k=20,
    filters=filters,
)

source_nodes = retriever.retrieve("What did the author do growing up?")

logger.debug(len(source_nodes))

logger.debug(source_nodes[0].text)
logger.debug(source_nodes[0].metadata)

logger.info("\n\n[DONE]", bright=True)