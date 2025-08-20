from dotenv import load_dotenv
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.chat_engine.types import ChatMessage
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.contextual import Contextual
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
# Contextual GLM
"""
logger.info("# Contextual GLM")

# %pip install llama-index-llms-contextual


load_dotenv()
llm = Contextual(model="v1", api_key=os.getenv("CONTEXTUAL_API_KEY"))

llm.complete(
    "Explain the importance of Grounded Language Models.",
    temperature=0.5,
    max_tokens=1024,
    top_p=0.9,
    avoid_commentary=False,
    knowledge=[
        "Contextual's Grounded Language Model (GLM) is the most grounded language model in the world. With state-of-the-art performance on FACTS (the leading groundedness benchmark) and our customer datasets, the GLM is the single best language model for RAG and agentic use cases for which minimizing hallucinations is critical. You can trust that the GLM will stick to the knowledge sources you give it"
    ],
)


llm.chat([ChatMessage(role="user", content="what color is the sky?")])

logger.info("\n\n[DONE]", bright=True)