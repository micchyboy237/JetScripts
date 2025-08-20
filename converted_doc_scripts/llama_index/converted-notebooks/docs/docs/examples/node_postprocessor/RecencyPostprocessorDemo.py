from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core import SummaryIndex
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import (
FixedRecencyPostprocessor,
EmbeddingRecencyPostprocessor,
)
from llama_index.core.response.notebook_utils import display_response
from llama_index.core.settings import Settings
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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
# Recency Filtering

Showcase capabilities of recency-weighted node postprocessor
"""
logger.info("# Recency Filtering")


# os.environ["OPENAI_API_KEY"] = "sk-..."


"""
### Parse Documents into Nodes, add to Docstore

In this example, there are 3 different versions of PG's essay. They are largely identical **except** 
for one specific section, which details the amount of funding they raised for Viaweb. 

V1: 50k, V2: 30k, V3: 10K

V1: 2020-01-01, V2: 2020-02-03, V3: 2022-04-12

The idea is to encourage index to fetch the most recent info (which is V3)
"""
logger.info("### Parse Documents into Nodes, add to Docstore")



def get_file_metadata(file_name: str):
    """Get file metadata."""
    if "v1" in file_name:
        return {"date": "2020-01-01"}
    elif "v2" in file_name:
        return {"date": "2020-02-03"}
    elif "v3" in file_name:
        return {"date": "2022-04-12"}
    else:
        raise ValueError("invalid file")


documents = SimpleDirectoryReader(
    input_files=[
        "test_versioned_data/paul_graham_essay_v1.txt",
        "test_versioned_data/paul_graham_essay_v2.txt",
        "test_versioned_data/paul_graham_essay_v3.txt",
    ],
    file_metadata=get_file_metadata,
).load_data()


Settings.text_splitter = SentenceSplitter(chunk_size=512)

nodes = Settings.text_splitter.get_nodes_from_documents(documents)

docstore = SimpleDocumentStore()
docstore.add_documents(nodes)

storage_context = StorageContext.from_defaults(docstore=docstore)

logger.debug(documents[2].get_text())

"""
### Build Index
"""
logger.info("### Build Index")

index = VectorStoreIndex(nodes, storage_context=storage_context)

"""
### Define Recency Postprocessors
"""
logger.info("### Define Recency Postprocessors")

node_postprocessor = FixedRecencyPostprocessor()

node_postprocessor_emb = EmbeddingRecencyPostprocessor()

"""
### Query Index
"""
logger.info("### Query Index")

query_engine = index.as_query_engine(
    similarity_top_k=3,
)
response = query_engine.query(
    "How much did the author raise in seed funding from Idelle's husband"
    " (Julian) for Viaweb?",
)

query_engine = index.as_query_engine(
    similarity_top_k=3, node_postprocessors=[node_postprocessor]
)
response = query_engine.query(
    "How much did the author raise in seed funding from Idelle's husband"
    " (Julian) for Viaweb?",
)

query_engine = index.as_query_engine(
    similarity_top_k=3, node_postprocessors=[node_postprocessor_emb]
)
response = query_engine.query(
    "How much did the author raise in seed funding from Idelle's husband"
    " (Julian) for Viaweb?",
)

"""
### Query Index (Lower-Level Usage)

In this example we first get the full set of nodes from a query call, and then send to node postprocessor, and then
finally synthesize response through a summary index.
"""
logger.info("### Query Index (Lower-Level Usage)")


query_str = (
    "How much did the author raise in seed funding from Idelle's husband"
    " (Julian) for Viaweb?"
)

query_engine = index.as_query_engine(
    similarity_top_k=3, response_mode="no_text"
)
init_response = query_engine.query(
    query_str,
)
resp_nodes = [n.node for n in init_response.source_nodes]

summary_index = SummaryIndex(resp_nodes)
query_engine = summary_index.as_query_engine(
    node_postprocessors=[node_postprocessor]
)
response = query_engine.query(query_str)

logger.info("\n\n[DONE]", bright=True)