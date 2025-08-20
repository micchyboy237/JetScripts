from jet.llm.mlx.base import MLX
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import IndexNode
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.experimental.query_engine import PandasQueryEngine
from typing import List
import camelot
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# Recursive Retriever + Query Engine Demo 

In this demo, we walk through a use case of showcasing our "RecursiveRetriever" module over hierarchical data.

The concept of recursive retrieval is that we not only explore the directly most relevant nodes, but also explore
node relationships to additional retrievers/query engines and execute them. For instance, a node may represent a concise summary of a structured table,
and link to a SQL/Pandas query engine over that structured table. Then if the node is retrieved, we want to also query the underlying query engine for the answer.

This can be especially useful for documents with hierarchical relationships. In this example, we walk through a Wikipedia article about billionaires (in PDF form), which contains both text and a variety of embedded structured tables. We first create a Pandas query engine over each table, but also represent each table by an `IndexNode` (stores a link to the query engine); this Node is stored along with other Nodes in a vector store. 

During query-time, if an `IndexNode` is fetched, then the underlying query engine/retriever will be queried. 

**Notes about Setup**

We use `camelot` to extract text-based tables from PDFs.
"""
logger.info("# Recursive Retriever + Query Engine Demo")

# %pip install llama-index-embeddings-ollama
# %pip install llama-index-readers-file pymupdf
# %pip install llama-index-llms-ollama
# %pip install llama-index-experimental



# from llama_index.readers.file import PyMuPDFReader

"""
## Default Settings
"""
logger.info("## Default Settings")


# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"


Settings.llm = MLX(model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats")
Settings.embed_model = MLXEmbedding(model="mxbai-embed-large")

"""
## Load in Document (and Tables)

# We use our `PyMuPDFReader` to read in the main text of the document.

We also use `camelot` to extract some structured tables from the document
"""
logger.info("## Load in Document (and Tables)")

file_path = "billionaires_page.pdf"

# reader = PyMuPDFReader()

docs = reader.load(file_path)

def get_tables(path: str, pages: List[int]):
    table_dfs = []
    for page in pages:
        table_list = camelot.read_pdf(path, pages=str(page))
        table_df = table_list[0].df
        table_df = (
            table_df.rename(columns=table_df.iloc[0])
            .drop(table_df.index[0])
            .reset_index(drop=True)
        )
        table_dfs.append(table_df)
    return table_dfs

table_dfs = get_tables(file_path, pages=[3, 25])

table_dfs[0]

table_dfs[1]

"""
## Create Pandas Query Engines

We create a pandas query engine over each structured table.

These can be executed on their own to answer queries about each table.

**WARNING:** This tool provides the LLM access to the `eval` function.
Arbitrary code execution is possible on the machine running this tool.
While some level of filtering is done on code, this tool is not recommended 
to be used in a production setting without heavy sandboxing or virtual machines.
"""
logger.info("## Create Pandas Query Engines")

llm = MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats")

df_query_engines = [
    PandasQueryEngine(table_df, llm=llm) for table_df in table_dfs
]

response = df_query_engines[0].query(
    "What's the net worth of the second richest billionaire in 2023?"
)
logger.debug(str(response))

response = df_query_engines[1].query(
    "How many billionaires were there in 2009?"
)
logger.debug(str(response))

"""
## Build Vector Index

Build vector index over the chunked document as well as over the additional `IndexNode` objects linked to the tables.
"""
logger.info("## Build Vector Index")


doc_nodes = Settings.node_parser.get_nodes_from_documents(docs)

summaries = [
    (
        "This node provides information about the world's richest billionaires"
        " in 2023"
    ),
    (
        "This node provides information on the number of billionaires and"
        " their combined net worth from 2000 to 2023."
    ),
]

df_nodes = [
    IndexNode(text=summary, index_id=f"pandas{idx}")
    for idx, summary in enumerate(summaries)
]

df_id_query_engine_mapping = {
    f"pandas{idx}": df_query_engine
    for idx, df_query_engine in enumerate(df_query_engines)
}

vector_index = VectorStoreIndex(doc_nodes + df_nodes)
vector_retriever = vector_index.as_retriever(similarity_top_k=1)

"""
## Use `RecursiveRetriever` in our `RetrieverQueryEngine`

We define a `RecursiveRetriever` object to recursively retrieve/query nodes. We then put this in our `RetrieverQueryEngine` along with a `ResponseSynthesizer` to synthesize a response.

We pass in mappings from id to retriever and id to query engine. We then pass in a root id representing the retriever we query first.
"""
logger.info("## Use `RecursiveRetriever` in our `RetrieverQueryEngine`")

vector_index0 = VectorStoreIndex(doc_nodes)
vector_query_engine0 = vector_index0.as_query_engine()


recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever},
    query_engine_dict=df_id_query_engine_mapping,
    verbose=True,
)

response_synthesizer = get_response_synthesizer(response_mode="compact")

query_engine = RetrieverQueryEngine.from_args(
    recursive_retriever, response_synthesizer=response_synthesizer
)

response = query_engine.query(
    "What's the net worth of the second richest billionaire in 2023?"
)

response.source_nodes[0].node.get_content()

str(response)

response = query_engine.query("How many billionaires were there in 2009?")

str(response)

response = vector_query_engine0.query(
    "How many billionaires were there in 2009?"
)

logger.debug(response.source_nodes[0].node.get_content())

logger.debug(str(response))

response.source_nodes[0].node.get_content()

response = query_engine.query(
    "Which billionaires are excluded from this list?"
)

logger.debug(str(response))

logger.info("\n\n[DONE]", bright=True)