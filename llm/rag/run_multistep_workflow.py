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

from jet.llm.query import MultiStepQueryEngineWorkflow
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


if __name__ == "__main__":
    # !mkdir -p 'mocks/paul_graham/'
    # !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'mocks/paul_graham/paul_graham_essay.txt'

    logger.debug("Loading documents...")
    documents = SimpleDirectoryReader("mocks/paul_graham").load_data()

    logger.debug("Loading llm...")
    settings = SettingsDict(
        llm_model="llama3.1",
        embedding_model="mxbai-embed-large",
        chunk_size=512,
        chunk_overlap=50,
        base_url="http://localhost:11434",
    )
    settings_manager = SettingsManager.create(settings)

    logger.debug("Creating nodes...")
    all_nodes = settings_manager.node_parser.get_nodes_from_documents(
        documents, show_progress=True)

    logger.debug("Loading index...")
    index = VectorStoreIndex(
        embed_model=settings_manager.embed_model,
        nodes=all_nodes,
        show_progress=True
    )

    logger.debug("Loading query engine...")
    query_engine = index.as_query_engine()

    logger.debug("Loading multi step workflow...")
    w = MultiStepQueryEngineWorkflow(timeout=200)

    # Sets maximum number of steps taken to answer the query.
    num_steps = 3

    # Set summary of the index, useful to create modified query at each step.
    index_summary = "Used to answer questions about the author"

    query = "In which city did the author found his first company, Viaweb?"

    async def run_workflow(query: str):
        result = await w.run(
            query=query,
            query_engine=query_engine,
            index_summary=index_summary,
            num_steps=num_steps,
        )

        # If created query in a step is None, the process will be stopped.

        display(
            Markdown("> Question: {}".format(query)),
            Markdown("Answer: {}".format(result)),
        )
        logger.log("> Question:\n", query, colors=["LOG", "DEBUG"])
        logger.log("Answer:\n", result, colors=["LOG", "SUCCESS"])
        sub_qa = result.metadata["sub_qa"]
        tuples = [(t[0], t[1].response) for t in sub_qa]
        return tuples

    import asyncio
    logger.debug("Running workflow...")
    tuples = asyncio.run(run_workflow(query=query))
    display(Markdown(f"{tuples}"))
    logger.log("Results:\n", tuples, colors=["LOG", "SUCCESS"])
