import joblib
import json
import os
import shutil

import requests
from pathlib import Path
from tqdm import tqdm
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import IndexNode
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core import StorageContext
from IPython.display import Markdown, display
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from jet.llm.ollama.base import Ollama
from llama_index.core import SummaryIndex
from llama_index.core import SimpleDirectoryReader
import sys
import logging
import nest_asyncio
from jet.llm.utils import display_jet_source_nodes
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/retrievers/auto_vs_recursive_retriever.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Comparing Methods for Structured Retrieval (Auto-Retrieval vs. Recursive Retrieval)
#
# In a naive RAG system, the set of input documents are then chunked, embedded, and dumped to a vector database collection. Retrieval would just fetch the top-k documents by embedding similarity.
#
# This can fail if the set of documents is large - it can be hard to disambiguate raw chunks, and you're not guaranteed to filter for the set of documents that contain relevant context.
#
# In this guide we explore **structured retrieval** - more advanced query algorithms that take advantage of structure within your documents for higher-precision retrieval. We compare the following two methods:
#
# - **Metadata Filters + Auto-Retrieval**: Tag each document with the right set of metadata. During query-time, use auto-retrieval to infer metadata filters along with passing through the query string for semantic search.
# - **Store Document Hierarchies (summaries -> raw chunks) + Recursive Retrieval**: Embed document summaries and map that to the set of raw chunks for each document. During query-time, do recursive retrieval to first fetch summaries before fetching documents.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

# %pip install llama-index-llms-ollama
# %pip install llama-index-vector-stores-weaviate

# !pip install llama-index


nest_asyncio.apply()


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

CACHE_DIR = Path("generated") / os.path.basename(__file__).split(".")[0]
SUMMARY_NODES_CACHE = Path(CACHE_DIR) / "summary_nodes.pkl"
VECTOR_RETRIEVERS_CACHE = Path(CACHE_DIR) / "vector_retrievers.pkl"

# Ensure the cache directory exists
Path(CACHE_DIR).mkdir(exist_ok=True)

# Custom settings
input_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
chunk_size = 512
chunk_overlap = 50

llm = Ollama(model="mistral", request_timeout=300.0, context_window=4096)
callback_manager = CallbackManager([LlamaDebugHandler()])
splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

# Setup output dir
base_dir = input_dir.split("/")[-1]
out_dir = Path("summaries") / base_dir
# Reset out_dir if it exists
if out_dir.exists():
    shutil.rmtree(out_dir)
# Create a new empty out_dir
out_dir.mkdir(parents=True, exist_ok=True)

# Read rag files
documents = SimpleDirectoryReader(input_dir, required_exts=[".md"]).load_data()
texts = [doc.text for doc in documents]

combined_file_path = os.path.join(input_dir, "combined.txt")
with open(combined_file_path, "w") as f:
    f.write("\n\n\n".join(texts))

include_files = [
    ".md",
    # "combined.txt",
]  # Add filenames to include here
exclude_files = []  # Add patterns or filenames to exclude here

rag_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]

# Apply include_files filter
if include_files:
    rag_files = [
        file for file in rag_files
        if any(include in file for include in include_files)
    ]

# Apply exclude_files filter
if exclude_files:
    rag_files = [
        file for file in rag_files
        if not any(exclude in file for exclude in exclude_files)
    ]

# Print the filtered list
logger.log("rag_files:", len(rag_files), colors=["WHITE", "DEBUG"])
logger.debug(json.dumps(rag_files, indent=2))


data_path = Path(input_dir)


wiki_titles = []
wiki_metadatas = {}

for file in rag_files:
    with open(file) as f:
        content = f.read()
        file_name = os.path.basename(file)

        title = file_name
        wiki_titles.append(title)
        wiki_metadatas[title] = {
            "file_name": file_name,
            "file_path": file
        }

docs_dict = {}
for wiki_title in wiki_titles:
    file_path = Path(wiki_metadatas[wiki_title]['file_path']).resolve()
    if not str(input_dir) in wiki_metadatas[wiki_title]['file_path']:
        continue

    doc = SimpleDirectoryReader(
        input_files=[file_path],
    ).load_data()[0]
    doc.metadata.update(wiki_metadatas[wiki_title])
    docs_dict[wiki_title] = doc


def build_recursive_retriever_over_document_summaries(similarity_top_k=3):
    # Build Recursive Retriever over Document Summaries
    summary_nodes = []
    vector_query_engines = {}
    vector_retrievers = {}

    # Filter wiki_titles to only those without existing summaries
    existing_summaries = set(
        p.stem for p in Path("summaries").rglob("*.txt")
    )
    titles_to_process = [
        title for title in wiki_titles if title not in existing_summaries]

    with tqdm(titles_to_process, total=len(titles_to_process)) as pbar:
        for wiki_title in pbar:
            # Update the description with the current wiki title
            pbar.set_description(f"Processing: {wiki_title}")

            # Build vector index and retriever for the title
            vector_index = VectorStoreIndex.from_documents(
                [docs_dict[wiki_title]],
                transformations=[splitter],
                callback_manager=callback_manager,
            )
            vector_query_engine = vector_index.as_query_engine(llm=llm)
            vector_query_engines[wiki_title] = vector_query_engine
            vector_retrievers[wiki_title] = vector_index.as_retriever(
                similarity_top_k=similarity_top_k
            )

            # Generate summary
            summary_index = SummaryIndex.from_documents(
                documents=[docs_dict[wiki_title]],
                callback_manager=callback_manager,
                show_progress=True,
            )
            summarizer = summary_index.as_query_engine(
                response_mode="tree_summarize", llm=llm
            )

            logger.newline()
            logger.log("Summary for", wiki_title,
                       "...", colors=["WHITE", "INFO"])
            response = summarizer.query(
                f"Summarize the contents of this document.")

            logger.log(
                "Summary nodes (tree_summarize):",
                f"({len(response.source_nodes)})",
                colors=["WHITE", "SUCCESS"]
            )
            display_jet_source_nodes(wiki_title, response.source_nodes)

            wiki_summary = response.response
            with open(out_dir / wiki_title, "a") as fp:
                fp.write(wiki_summary)

            node = IndexNode(text=wiki_summary, index_id=wiki_title)
            summary_nodes.append(node)

    logger.log("Summary nodes:", len(summary_nodes),
               colors=["WHITE", "SUCCESS"])

    return summary_nodes, vector_retrievers


@time_it
def query_nodes(query, nodes, vector_retrievers, similarity_top_k=3):
    logger.debug(f"Querying ({len(nodes)}) nodes...")
    top_vector_index = VectorStoreIndex(
        nodes, transformations=[splitter], callback_manager=callback_manager
    )
    top_vector_retriever = top_vector_index.as_retriever(
        similarity_top_k=similarity_top_k)

    recursive_retriever = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": top_vector_retriever, **vector_retrievers},
        verbose=False,
    )

    retrieved_nodes = recursive_retriever.retrieve(query)

    # Sort retrieved_nodes by item.score in reverse order
    retrieved_nodes_sorted = sorted(
        retrieved_nodes, key=lambda item: item.score, reverse=True)

    logger.log("Query:", query, colors=["WHITE", "INFO"])
    logger.log(
        "Retrieved summary nodes (RecursiveRetriever):",
        f"({len(retrieved_nodes_sorted)})",
        colors=["WHITE", "SUCCESS"]
    )

    return retrieved_nodes_sorted


def load_from_cache_or_compute(use_cache=False, similarity_top_k=3):
    """Load cached data or compute if not available."""
    if use_cache and SUMMARY_NODES_CACHE.exists() and VECTOR_RETRIEVERS_CACHE.exists():
        summary_nodes = joblib.load(SUMMARY_NODES_CACHE)
        vector_retrievers = joblib.load(VECTOR_RETRIEVERS_CACHE)
        logger.success("Cache hit! Loaded data.")
    else:
        logger.debug("Cache not found. Building data...")
        summary_nodes, vector_retrievers = build_recursive_retriever_over_document_summaries(
            similarity_top_k=similarity_top_k)
        joblib.dump(summary_nodes, SUMMARY_NODES_CACHE)
        joblib.dump(vector_retrievers, VECTOR_RETRIEVERS_CACHE)
        logger.success("Data cached successfully.")

    return summary_nodes, vector_retrievers


if __name__ == "__main__":
    use_cache = False
    similarity_top_k = 4

    # main_metadata_filters_and_auto_retrieval()
    logger.debug("Building recursive retriever over document summaries...")
    summary_nodes, vector_retrievers = load_from_cache_or_compute(
        use_cache=use_cache,
        similarity_top_k=similarity_top_k,
    )

    # Sample usage
    query_top_k = len(summary_nodes)

    query = "Tell me about yourself."
    retrieved_nodes = query_nodes(
        query,
        summary_nodes,
        vector_retrievers,
        similarity_top_k=query_top_k,
    )
    display_jet_source_nodes(query, retrieved_nodes)

    retrieved_contents = []
    for node in retrieved_nodes:
        retrieved_contents.append(node.node.get_content())

    result = "\n\n".join(retrieved_contents)
    # logger.newline()
    # logger.log("Final result:")
    # logger.success(result)

    # Run app
    while True:
        # Continuously ask user for queries
        try:
            query = input("Enter your query (type 'exit' to quit): ").strip()
            if query.lower() == "exit":
                print("Exiting query loop.")
                break

            retrieved_nodes = query_nodes(
                query,
                summary_nodes,
                vector_retrievers,
                similarity_top_k=query_top_k,
            )
            display_jet_source_nodes(query, retrieved_nodes)

            result = "\n\n".join(retrieved_contents)
            # logger.newline()
            # logger.log("Final result:")
            # logger.success(result)

        except KeyboardInterrupt:
            print("\nExiting query loop.")
            break
        except Exception as e:
            logger.error(f"Error while processing query: {e}")
