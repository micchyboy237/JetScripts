from IPython.display import Markdown, display
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core import Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import VectorStoreIndex
from llama_index.packs.longrag import LongRAGPack
import os
import shutil
import typing as t


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# LongRAG example

This LlamaPack implements LongRAG based on [this paper](https://arxiv.org/pdf/2406.15319).

LongRAG retrieves large tokens at a time, with each retrieval unit being ~6k tokens long, consisting of entire documents or groups of documents. This contrasts the short retrieval units (100 word passages) of traditional RAG. LongRAG is advantageous because results can be achieved using only the top 4-8 retrieval units, and long-context LLMs can better understand the context of the documents because long retrieval units preserve their semantic integrity.

## Setup
"""
logger.info("# LongRAG example")

# import nest_asyncio

# nest_asyncio.apply()

# %pip install llama-index


# os.environ["OPENAI_API_KEY"] = "<Your API Key>"

"""
## Usage

Below shows the usage of `LongRAGPack` using the `gpt-4o` LLM, which is able to handle long context inputs.
"""
logger.info("## Usage")


Settings.llm = OllamaFunctionCallingAdapter("gpt-4o")

pack = LongRAGPack(
    data_dir="/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp")


query_str = (
    "How can Pittsburgh become a startup hub, and what are the two types of moderates?"
)
res = pack.run(query_str)
display(Markdown(str(res)))

"""
Other parameters include `chunk_size`, `similarity_top_k`, and `small_chunk_size`.
- `chunk_size`: To demonstrate how different documents are grouped together, documents are split into nodes of `chunk_size` tokens, then re-grouped based on the relationships between the nodes. Because this does not affect the final answer, it can be disabled by setting `chunk_size` to None. The default size is 4096.
- `similarity_top_k`: Retrieves the top k large retrieval units. The default is 8, and based on the paper, the ideal range is 4-8.
- `small_chunk_size`: To compare similarities, each large retrieval unit is split into smaller child retrieval units of `small_chunk_size` tokens. The embeddings of these smaller retrieval units are compared to the query embeddings. The top k large parent retrieval units are chosen based on the maximum scores of their smaller child retrieval units. The default size is 512.
"""
logger.info(
    "Other parameters include `chunk_size`, `similarity_top_k`, and `small_chunk_size`.")

pack = LongRAGPack(
    data_dir="/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp", chunk_size=None, similarity_top_k=4)
query_str = (
    "How can Pittsburgh become a startup hub, and what are the two types of moderates?"
)
res = pack.run(query_str)
display(Markdown(str(res)))

"""
## Vector Storage

The vector index can be extracted and be persisted to disk. A `LongRAGPack` can also be constructed given a vector index. Below is an example of persisting the index to disk.
"""
logger.info("## Vector Storage")


modules = pack.get_modules()
index = t.cast(VectorStoreIndex, modules["index"])
index.storage_context.persist(persist_dir="./paul_graham")

"""
Below is an example of loading an index.
"""
logger.info("Below is an example of loading an index.")


ctx = StorageContext.from_defaults(persist_dir="./paul_graham")
index = load_index_from_storage(ctx)
pack_from_idx = LongRAGPack(
    data_dir="/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp", index=index)
query_str = (
    "How can Pittsburgh become a startup hub, and what are the two types of moderates?"
)
res = pack.run(query_str)
display(Markdown(str(res)))

logger.info("\n\n[DONE]", bright=True)
