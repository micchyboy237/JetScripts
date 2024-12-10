from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)
from llama_index.core.response_synthesizers import (
    get_response_synthesizer,
)

from llama_index.core.schema import QueryBundle, TextNode

from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)

from llama_index.core import Settings
from llama_index.core.llms import LLM

from typing import TypedDict, cast
from IPython.display import Markdown, display

from llama_index.core.workflow import Event
from typing import Dict, List, Any
from llama_index.core.schema import NodeWithScore


from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex
from llama_index.core.callbacks import CallbackManager

from jet.llm.query import LongRAGWorkflow
from jet.logger import logger


class SettingsDict(TypedDict):
    llm_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    base_url: str


class OllamaCallbackManager(CallbackManager):
    def on_event_start(
        self,
        *args,
        **kwargs: Any,
    ):
        logger.log("OllamaCallbackManager on_event_start:", {
                   **args, **kwargs}, colors=["LOG", "INFO"])

    def on_event_end(
        self,
        *args,
        **kwargs: Any,
    ):
        logger.log("OllamaCallbackManager on_event_end:", {
                   **args, **kwargs}, colors=["LOG", "INFO"])


class SettingsManager:
    @staticmethod
    def create(settings: SettingsDict):
        Settings.chunk_size = settings["chunk_size"]
        Settings.chunk_overlap = settings["chunk_overlap"]
        Settings.embed_model = OllamaEmbedding(
            model_name=settings["embedding_model"],
            base_url=settings["base_url"],
            callback_manager=OllamaCallbackManager(),
        )
        Settings.llm = Ollama(
            temperature=0,
            request_timeout=120.0,
            model=settings["llm_model"],
            base_url=settings["base_url"],
        )
        return Settings

# Main Function


async def main():
    from jet.logger import logger

    logger.debug("Loading llm...")
    settings = SettingsDict(
        llm_model="llama3.1",
        embedding_model="mxbai-embed-large",
        chunk_size=512,
        chunk_overlap=50,
        base_url="http://localhost:11434",
    )
    settings_manager = SettingsManager.create(settings)
    wf = LongRAGWorkflow(timeout=60)

    data_dir = "data"
    result = await wf.run(data_dir=data_dir, llm=llm, chunk_size=DEFAULT_CHUNK_SIZE, similarity_top_k=DEFAULT_TOP_K, small_chunk_size=DEFAULT_SMALL_CHUNK_SIZE)
    res = await wf.run(query_str="How can Pittsburgh become a startup hub?", query_eng=result["query_engine"])
    print(res)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
