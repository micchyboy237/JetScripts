import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.bridge.pydantic import Field
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.span import BaseSpan
from llama_index.core.instrumentation.span.base import BaseSpan
from llama_index.core.instrumentation.span_handlers import BaseSpanHandler
from llama_index.core.instrumentation.span_handlers import SimpleSpanHandler
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import Any, Dict, Optional
import inspect
import llama_index.core.instrumentation as instrument
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
# Instrumentation: Basic Usage

The `instrumentation` module can be used for observability and monitoring of your llama-index application. It is comprised of the following core abstractions:

- `Event` — represents a single moment in time that a certain occurrence took place within the execution of the application’s code.
- `EventHandler` — listen to the occurrences of `Event`'s and execute code logic at these moments in time.
- `Span` — represents the execution flow of a particular part in the application’s code and thus contains `Event`'s.
- `SpanHandler` — is responsible for the entering, exiting, and dropping (i.e., early exiting due to error) of `Span`'s.
- `Dispatcher` — emits `Event`'s as well as signals to enter/exit/drop a `Span` to the appropriate handlers.

In this notebook, we demonstrate the basic usage pattern of `instrumentation`:

1. Define your custom `EventHandler`
2. Define your custom `SpanHandler` which handles an associated `Span` type
3. Attach your `EventHandler` and `SpanHandler` to the dispatcher of choice (here, we'll attach it to the root dispatcher).
"""
logger.info("# Instrumentation: Basic Usage")

# import nest_asyncio

# nest_asyncio.apply()

"""
### Custom Event Handlers
"""
logger.info("### Custom Event Handlers")


"""
Defining your custom `EventHandler` involves subclassing the `BaseEventHandler`. Doing so, requires defining logic for the abstract method `handle()`.
"""
logger.info("Defining your custom `EventHandler` involves subclassing the `BaseEventHandler`. Doing so, requires defining logic for the abstract method `handle()`.")

class MyEventHandler(BaseEventHandler):
    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "MyEventHandler"

    def handle(self, event) -> None:
        """Logic for handling event."""
        logger.debug(event.dict())
        logger.debug("")
        with open("log.txt", "a") as f:
            f.write(str(event))
            f.write("\n")

"""
### Custom Span Handlers

`SpanHandler` also involve subclassing a base class, in this case `BaseSpanHandler`. However, since `SpanHandler`'s work with an associated `Span` type, you will need to create this as well if you want to handle a new `Span` type.
"""
logger.info("### Custom Span Handlers")




class MyCustomSpan(BaseSpan):
    custom_field_1: Any = Field(...)
    custom_field_2: Any = Field(...)


class MyCustomSpanHandler(BaseSpanHandler[MyCustomSpan]):
    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "MyCustomSpanHandler"

    def new_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[MyCustomSpan]:
        """Create a span."""
        pass

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Logic for preparing to exit a span."""
        pass

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> Any:
        """Logic for preparing to drop a span."""
        pass

"""
For this notebook, we'll use `SimpleSpanHandler` that works with the `SimpleSpan` type.
"""
logger.info("For this notebook, we'll use `SimpleSpanHandler` that works with the `SimpleSpan` type.")


"""
### Dispatcher

Now that we have our `EventHandler` and our `SpanHandler`, we can attach it to a `Dispatcher` that will emit `Event`'s and signals to start/exit/drop a `Span` to the appropriate handlers. Those that are familiar with `Logger` from the `logging` Python module, might notice that `Dispatcher` adopts a similar interface. What's more is that `Dispatcher` also utilizes a similar hierarchy and propagation scheme as `Logger`. Specifically, a `dispatcher` will emit `Event`'s to its handlers and by default propagate these events to its parent `Dispatcher` for it to send to its own handlers.
"""
logger.info("### Dispatcher")


dispatcher = instrument.get_dispatcher()  # modify root dispatcher

span_handler = SimpleSpanHandler()

dispatcher.add_event_handler(MyEventHandler())
dispatcher.add_span_handler(span_handler)

"""
You can also get dispatcher's by name. Purely for the sake of demonstration, in the cells below we get the dispatcher that is defined in the `base.base_query_engine` submodule of `llama_index.core`.
"""
logger.info("You can also get dispatcher's by name. Purely for the sake of demonstration, in the cells below we get the dispatcher that is defined in the `base.base_query_engine` submodule of `llama_index.core`.")

qe_dispatcher = instrument.get_dispatcher(
    "llama_index.core.base.base_query_engine"
)

qe_dispatcher

qe_dispatcher.parent

"""
### Test It Out
"""
logger.info("### Test It Out")

# !mkdir -p 'data/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'


documents = SimpleDirectoryReader(input_dir="./data").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

"""
#### Sync
"""
logger.info("#### Sync")

query_result = query_engine.query("Who is Paul?")

"""
#### Async

`Dispatcher` also works on async methods.
"""
logger.info("#### Async")

async def run_async_code_875e2e24():
    async def run_async_code_cbdd119a():
        query_result = query_engine.query("Who is Paul?")
        return query_result
    query_result = asyncio.run(run_async_code_cbdd119a())
    logger.success(format_json(query_result))
    return query_result
query_result = asyncio.run(run_async_code_875e2e24())
logger.success(format_json(query_result))

"""
#### Streaming

`Dispatcher` also works on methods that support streaming!
"""
logger.info("#### Streaming")

chat_engine = index.as_chat_engine()

streaming_response = chat_engine.stream_chat("Tell me a joke.")

for token in streaming_response.response_gen:
    logger.debug(token, end="")

"""
### Printing Basic Trace Trees with `SimpleSpanHandler`
"""
logger.info("### Printing Basic Trace Trees with `SimpleSpanHandler`")

span_handler.print_trace_trees()

logger.info("\n\n[DONE]", bright=True)