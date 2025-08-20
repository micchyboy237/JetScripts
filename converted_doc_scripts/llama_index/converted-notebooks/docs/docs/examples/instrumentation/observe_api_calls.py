from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events.embedding import EmbeddingEndEvent
from llama_index.core.instrumentation.events.llm import (
LLMCompletionEndEvent,
LLMChatEndEvent,
)
from llama_index.core.settings import Settings
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
# API Call Observability 

Using the new `instrumentation` package, we can get direct observability into API calls made using LLMs and emebdding models.

In this notebook, we explore doing this in order to add observability to LLM and embedding calls.
"""
logger.info("# API Call Observability")


# os.environ["OPENAI_API_KEY"] = "sk-..."

"""
## Defining an Event Handler
"""
logger.info("## Defining an Event Handler")



class ModelEventHandler(BaseEventHandler):
    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "ModelEventHandler"

    def handle(self, event) -> None:
        """Logic for handling event."""
        if isinstance(event, LLMCompletionEndEvent):
            logger.debug(f"LLM Prompt length: {len(event.prompt)}")
            logger.debug(f"LLM Completion: {str(event.response.text)}")
        elif isinstance(event, LLMChatEndEvent):
            messages_str = "\n".join([str(x) for x in event.messages])
            logger.debug(f"LLM Input Messages length: {len(messages_str)}")
            logger.debug(f"LLM Response: {str(event.response.message)}")
        elif isinstance(event, EmbeddingEndEvent):
            logger.debug(f"Embedding {len(event.chunks)} text chunks")

"""
## Attaching the Event Handler
"""
logger.info("## Attaching the Event Handler")


root_dispatcher = get_dispatcher()

root_dispatcher.add_event_handler(ModelEventHandler())

"""
## Invoke the Handler!
"""
logger.info("## Invoke the Handler!")


index = VectorStoreIndex.from_documents([Document.example()])

query_engine = index.as_query_engine()
response = query_engine.query("Tell me about LLMs?")

query_engine = index.as_query_engine(streaming=True)
response = query_engine.query("Repeat only these two words: Hello world!")
for r in response.response_gen:
    ...

logger.info("\n\n[DONE]", bright=True)