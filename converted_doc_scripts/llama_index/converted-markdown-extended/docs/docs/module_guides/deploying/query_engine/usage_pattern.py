from jet.logger import logger
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.retrievers import VectorIndexRetriever
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# Usage Pattern

## Get Started

Build a query engine from index:
"""
logger.info("# Usage Pattern")

query_engine = index.as_query_engine()

"""
<Aside type="tip">
To learn how to build an index, see [Indexing](/python/framework/module_guides/indexing)
</Aside>

Ask a question over your data
"""
logger.info("To learn how to build an index, see [Indexing](/python/framework/module_guides/indexing)")

response = query_engine.query("Who is Paul Graham?")

"""
## Configuring a Query Engine

### High-Level API

You can directly build and configure a query engine from an index in 1 line of code:
"""
logger.info("## Configuring a Query Engine")

query_engine = index.as_query_engine(
    response_mode="tree_summarize",
    verbose=True,
)

"""
> Note: While the high-level API optimizes for ease-of-use, it does _NOT_ expose full range of configurability.

See [**Response Modes**](/python/framework/module_guides/deploying/query_engine/response_modes) for a full list of response modes and what they do.

### Low-Level Composition API

You can use the low-level composition API if you need more granular control.
Concretely speaking, you would explicitly construct a `QueryEngine` object instead of calling `index.as_query_engine(...)`.

> Note: You may need to look at API references or example notebooks.
"""
logger.info("### Low-Level Composition API")


index = VectorStoreIndex.from_documents(documents)

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=2,
)

response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize",
)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

response = query_engine.query("What did the author do growing up?")
logger.debug(response)

"""
### Streaming

To enable streaming, you simply need to pass in a `streaming=True` flag
"""
logger.info("### Streaming")

query_engine = index.as_query_engine(
    streaming=True,
)
streaming_response = query_engine.query(
    "What did the author do growing up?",
)
streaming_response.print_response_stream()

"""
- Read the full [streaming guide](/python/framework/module_guides/deploying/query_engine/streaming)
- See an [end-to-end example](/python/examples/customization/streaming/simpleindexdemo-streaming)

## Defining a Custom Query Engine

You can also define a custom query engine. Simply subclass the `CustomQueryEngine` class, define any attributes you'd want to have (similar to defining a Pydantic class), and implement a `custom_query` function that returns either a `Response` object or a string.
"""
logger.info("## Defining a Custom Query Engine")



class RAGQueryEngine(CustomQueryEngine):
    """RAG Query Engine."""

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        response_obj = self.response_synthesizer.synthesize(query_str, nodes)
        return response_obj

"""
See the [Custom Query Engine guide](/python/examples/query_engine/custom_query_engine) for more details.
"""
logger.info("See the [Custom Query Engine guide](/python/examples/query_engine/custom_query_engine) for more details.")

logger.info("\n\n[DONE]", bright=True)