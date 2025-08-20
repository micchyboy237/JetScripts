from datetime import datetime, timedelta
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core import SummaryIndex
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import TimeWeightedPostprocessor
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
# Time-Weighted Rerank

Showcase capabilities of time-weighted node postprocessor
"""
logger.info("# Time-Weighted Rerank")


"""
### Parse Documents into Nodes, add to Docstore

In this example, there are 3 different versions of PG's essay. They are largely identical **except** 
for one specific section, which details the amount of funding they raised for Viaweb. 

V1: 50k, V2: 30k, V3: 10K

V1: -1 day, V2: -2 days, V3: -3 days

The idea is to encourage index to fetch the most recent info (which is V3)
"""
logger.info("### Parse Documents into Nodes, add to Docstore")



now = datetime.now()
key = "__last_accessed__"


doc1 = SimpleDirectoryReader(
    input_files=["./test_versioned_data/paul_graham_essay_v1.txt"]
).load_data()[0]


doc2 = SimpleDirectoryReader(
    input_files=["./test_versioned_data/paul_graham_essay_v2.txt"]
).load_data()[0]

doc3 = SimpleDirectoryReader(
    input_files=["./test_versioned_data/paul_graham_essay_v3.txt"]
).load_data()[0]



Settings.text_splitter = SentenceSplitter(chunk_size=512)

nodes1 = Settings.text_splitter.get_nodes_from_documents([doc1])
nodes2 = Settings.text_splitter.get_nodes_from_documents([doc2])
nodes3 = Settings.text_splitter.get_nodes_from_documents([doc3])


nodes1[14].metadata[key] = (now - timedelta(hours=3)).timestamp()
nodes1[14].excluded_llm_metadata_keys = [key]

nodes2[14].metadata[key] = (now - timedelta(hours=2)).timestamp()
nodes2[14].excluded_llm_metadata_keys = [key]

nodes3[14].metadata[key] = (now - timedelta(hours=1)).timestamp()
nodes2[14].excluded_llm_metadata_keys = [key]


docstore = SimpleDocumentStore()
nodes = [nodes1[14], nodes2[14], nodes3[14]]
docstore.add_documents(nodes)

storage_context = StorageContext.from_defaults(docstore=docstore)

"""
### Build Index
"""
logger.info("### Build Index")

index = VectorStoreIndex(nodes, storage_context=storage_context)

"""
### Define Recency Postprocessors
"""
logger.info("### Define Recency Postprocessors")

node_postprocessor = TimeWeightedPostprocessor(
    time_decay=0.5, time_access_refresh=False, top_k=1
)

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

display_response(response)

query_engine = index.as_query_engine(
    similarity_top_k=3, node_postprocessors=[node_postprocessor]
)
response = query_engine.query(
    "How much did the author raise in seed funding from Idelle's husband"
    " (Julian) for Viaweb?",
)

display_response(response)

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
resp_nodes = [n for n in init_response.source_nodes]

new_resp_nodes = node_postprocessor.postprocess_nodes(resp_nodes)

summary_index = SummaryIndex([n.node for n in new_resp_nodes])
query_engine = summary_index.as_query_engine()
response = query_engine.query(query_str)

display_response(response)

logger.info("\n\n[DONE]", bright=True)