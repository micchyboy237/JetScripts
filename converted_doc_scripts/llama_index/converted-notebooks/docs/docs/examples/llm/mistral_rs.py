from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.mistral_rs import MistralRS
from mistralrs import Which, Architecture
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
# MistralRS LLM

**NOTE:** MistralRS requires a rust package manager called `cargo` to be installed. Visit https://rustup.rs/ for installation details.
"""
logger.info("# MistralRS LLM")

# %pip install llama-index-core
# %pip install llama-index-readers-file
# %pip install llama-index-llms-mistral-rs
# %pip install llama-index-llms-huggingface


documents = SimpleDirectoryReader("data").load_data()

Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

"""
MistralRS uses model IDs from huggingface hub.
"""
logger.info("MistralRS uses model IDs from huggingface hub.")

Settings.llm = MistralRS(
    which=Which.Plain(
        model_id="mistralai/Mistral-7B-Instruct-v0.1",
        arch=Architecture.Mistral,
        tokenizer_json=None,
        repeat_last_n=64,
    ),
    max_new_tokens=4096,
    context_window=1024 * 5,
)

Settings.llm = MistralRS(
    which=Which.GGUF(
        tok_model_id="mistralai/Mistral-7B-Instruct-v0.1",
        quantized_model_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        quantized_filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        tokenizer_json=None,
        repeat_last_n=64,
    ),
    max_new_tokens=4096,
    context_window=1024 * 5,
)

index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine()
response = query_engine.query("How do I pronounce graphene?")
logger.debug(response)

logger.info("\n\n[DONE]", bright=True)