from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_hub.semanticscholar.base import SemanticScholarReader
from llama_index import (
VectorStoreIndex,
StorageContext,
load_index_from_storage,
ServiceContext,
)
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms import MLX
from llama_index.query_engine import CitationQueryEngine
from llama_index.response.notebook_utils import display_response
import openai
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
# Semantic Scholar Loader in llama-index
"""
logger.info("# Semantic Scholar Loader in llama-index")


s2reader = SemanticScholarReader()

# openai.api_key = os.environ["OPENAI_API_KEY"]
service_context = ServiceContext.from_defaults(
    llm=MLX(model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats", temperature=0)
)

query_space = "large language models"
full_text = True
total_papers = 50

persist_dir = (
    "./citation_" + query_space + "_" + str(total_papers) + "_" + str(full_text)
)

if not os.path.exists(persist_dir):
    documents = s2reader.load_data(query_space, total_papers, full_text=full_text)
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    index.storage_context.persist(persist_dir=persist_dir)
else:
    index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir=persist_dir),
        service_context=service_context,
    )

query_engine = CitationQueryEngine.from_args(
    index,
    similarity_top_k=3,
    citation_chunk_size=512,
)

query_string = "limitations of using large language models"

response = query_engine.query(query_string)
display_response(
    response, show_source=True, source_length=100, show_source_metadata=True
)

query_space = "covid 19 vaccine"
query_string = "List the efficacy numbers of the covid 19 vaccines"
full_text = True
total_papers = 50

persist_dir = (
    "./citation_" + query_space + "_" + str(total_papers) + "_" + str(full_text)
)

if not os.path.exists(persist_dir):
    documents = s2reader.load_data(query_space, total_papers, full_text=full_text)
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    index.storage_context.persist(persist_dir=persist_dir)
else:
    index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir=persist_dir),
        service_context=service_context,
    )

query_engine = CitationQueryEngine.from_args(
    index,
    similarity_top_k=3,
    citation_chunk_size=512,
)

response = query_engine.query(query_string)
display_response(
    response, show_source=True, source_length=100, show_source_metadata=True
)

logger.info("\n\n[DONE]", bright=True)