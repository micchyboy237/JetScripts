from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import logging
import os
import shutil
import sys


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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/customization/streaming/chat_engine_condense_question_stream_response.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Streaming for Chat Engine - Condense Question Mode

Load documents, build the VectorStoreIndex
"""
logger.info("# Streaming for Chat Engine - Condense Question Mode")


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()

index = VectorStoreIndex.from_documents(documents)

"""
Chat with your data
"""
logger.info("Chat with your data")

chat_engine = index.as_chat_engine(
    chat_mode="condense_question", streaming=True
)
response_stream = chat_engine.stream_chat("What did Paul Graham do after YC?")

response_stream.print_response_stream()

"""
Ask a follow up question
"""
logger.info("Ask a follow up question")

response_stream = chat_engine.stream_chat("What about after that?")

response_stream.print_response_stream()

response_stream = chat_engine.stream_chat("Can you tell me more?")

response_stream.print_response_stream()

"""
Reset conversation state
"""
logger.info("Reset conversation state")

chat_engine.reset()

response_stream = chat_engine.stream_chat("What about after that?")

response_stream.print_response_stream()

logger.info("\n\n[DONE]", bright=True)