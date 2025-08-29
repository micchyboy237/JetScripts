from jet.models.config import MODELS_CACHE_DIR
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from time import time
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/node_postprocessor/SentenceTransformerRerank.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Rerank can speed up an LLM query without sacrificing accuracy (and in fact, probably improving it). It does so by pruning away irrelevant nodes from the context.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("Rerank can speed up an LLM query without sacrificing accuracy (and in fact, probably improving it). It does so by pruning away irrelevant nodes from the context.")

# %pip install llama-index-embeddings-huggingface
# %pip install llama-index-llms-ollama

# !pip install llama-index


"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()


Settings.llm = OllamaFunctionCallingAdapter(model="llama3.2", request_timeout=300.0, context_window=4096)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

index = VectorStoreIndex.from_documents(documents=documents)


rerank = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3
)

"""
First, we try with reranking. We time the query to see how long it takes to process the output from the retrieved context.
"""
logger.info("First, we try with reranking. We time the query to see how long it takes to process the output from the retrieved context.")


query_engine = index.as_query_engine(
    similarity_top_k=10, node_postprocessors=[rerank]
)

now = time()
response = query_engine.query(
    "Which grad schools did the author apply for and why?",
)
logger.debug(f"Elapsed: {round(time() - now, 2)}s")

logger.debug(response)

logger.debug(response.get_formatted_sources(length=200))

"""
Next, we try without rerank
"""
logger.info("Next, we try without rerank")

query_engine = index.as_query_engine(similarity_top_k=10)


now = time()
response = query_engine.query(
    "Which grad schools did the author apply for and why?",
)

logger.debug(f"Elapsed: {round(time() - now, 2)}s")

logger.debug(response)

logger.debug(response.get_formatted_sources(length=200))

"""
As we can see, the query engine with reranking produced a much more concise output in much lower time (4s v.s. 28s). While both responses were essentially correct, the query engine without reranking included a lot of irrelevant information - a phenomenon we could attribute to "pollution of the context window".
"""
logger.info("As we can see, the query engine with reranking produced a much more concise output in much lower time (4s v.s. 28s). While both responses were essentially correct, the query engine without reranking included a lot of irrelevant information - a phenomenon we could attribute to "pollution of the context window".")

logger.info("\n\n[DONE]", bright=True)