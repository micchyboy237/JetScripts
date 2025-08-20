from IPython.display import Markdown
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_hub.openalex import OpenAlexReader
from llama_index import (
VectorStoreIndex,
ServiceContext,
)
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms import MLX
from llama_index.query_engine import CitationQueryEngine
from llama_index.response.notebook_utils import display_response
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



openalex_reader = OpenAlexReader(email="shauryr@gmail.com")

query = "biases in large language models"

works = openalex_reader.load_data(query, full_text=False)
service_context = ServiceContext.from_defaults(
    llm=MLXLlamaIndexLLMAdapter(model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats", temperature=0)
)
index = VectorStoreIndex.from_documents(works, service_context=service_context)

query_engine = CitationQueryEngine.from_args(
    index,
    similarity_top_k=10,
    citation_chunk_size=1024,
)
response = query_engine.query(
    "list the biases in large language models in a markdown table"
)


Markdown(response.response)

display_response(
    response, show_source=True, source_length=100, show_source_metadata=True
)

logger.info("\n\n[DONE]", bright=True)