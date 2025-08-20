from jet.logger import CustomLogger
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.events.agent import (
AgentChatWithStepStartEvent,
AgentChatWithStepEndEvent,
AgentRunStepStartEvent,
AgentRunStepEndEvent,
AgentToolCallEvent,
)
from llama_index.core.instrumentation.events.chat_engine import (
StreamChatErrorEvent,
StreamChatDeltaReceivedEvent,
)
from llama_index.core.instrumentation.events.embedding import (
EmbeddingStartEvent,
EmbeddingEndEvent,
)
from llama_index.core.instrumentation.events.llm import (
LLMPredictEndEvent,
LLMPredictStartEvent,
LLMStructuredPredictEndEvent,
LLMStructuredPredictStartEvent,
LLMCompletionEndEvent,
LLMCompletionStartEvent,
LLMChatEndEvent,
LLMChatStartEvent,
LLMChatInProgressEvent,
)
from llama_index.core.instrumentation.events.query import (
QueryStartEvent,
QueryEndEvent,
)
from llama_index.core.instrumentation.events.rerank import (
ReRankStartEvent,
ReRankEndEvent,
)
from llama_index.core.instrumentation.events.retrieval import (
RetrievalStartEvent,
RetrievalEndEvent,
)
from llama_index.core.instrumentation.events.span import (
SpanDropEvent,
)
from llama_index.core.instrumentation.events.synthesis import (
SynthesizeStartEvent,
SynthesizeEndEvent,
GetResponseEndEvent,
GetResponseStartEvent,
)
from llama_index.core.instrumentation.span import SimpleSpan
from llama_index.core.instrumentation.span_handlers import SimpleSpanHandler
from llama_index.core.instrumentation.span_handlers.base import BaseSpanHandler
from treelib import Tree
from typing import Any, Optional
from typing import Dict, List
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Built-In Observability Instrumentation

Within LlamaIndex, many events and spans are created and logged through our instrumentation system.

This notebook walks through how you would hook into these events and spans to create your own observability tooling.
"""
logger.info("# Built-In Observability Instrumentation")

# %pip install llama-index treelib

"""
## Events

LlamaIndex logs several types of events. Events are singular data points that occur during runtime, and usually belong to some parent span.

Below is a thorough list of what is logged, and how to create an event handler to read these events.
"""
logger.info("## Events")





class ExampleEventHandler(BaseEventHandler):
    """Example event handler.

    This event handler is an example of how to create a custom event handler.

    In general, logged events are treated as single events in a point in time,
    that link to a span. The span is a collection of events that are related to
    a single task. The span is identified by a unique span_id.

    While events are independent, there is some hierarchy.
    For example, in query_engine.query() call with a reranker attached:
    - QueryStartEvent
    - RetrievalStartEvent
    - EmbeddingStartEvent
    - EmbeddingEndEvent
    - RetrievalEndEvent
    - RerankStartEvent
    - RerankEndEvent
    - SynthesizeStartEvent
    - GetResponseStartEvent
    - LLMPredictStartEvent
    - LLMChatStartEvent
    - LLMChatEndEvent
    - LLMPredictEndEvent
    - GetResponseEndEvent
    - SynthesizeEndEvent
    - QueryEndEvent
    """

    events: List[BaseEvent] = []

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "ExampleEventHandler"

    def handle(self, event: BaseEvent) -> None:
        """Logic for handling event."""
        logger.debug("-----------------------")
        logger.debug(event.id_)
        logger.debug(event.timestamp)
        logger.debug(event.span_id)

        logger.debug(f"Event type: {event.class_name()}")
        if isinstance(event, AgentRunStepStartEvent):
            logger.debug(event.task_id)
            logger.debug(event.step)
            logger.debug(event.input)
        if isinstance(event, AgentRunStepEndEvent):
            logger.debug(event.step_output)
        if isinstance(event, AgentChatWithStepStartEvent):
            logger.debug(event.user_msg)
        if isinstance(event, AgentChatWithStepEndEvent):
            logger.debug(event.response)
        if isinstance(event, AgentToolCallEvent):
            logger.debug(event.arguments)
            logger.debug(event.tool.name)
            logger.debug(event.tool.description)
            logger.debug(event.tool.to_openai_tool())
        if isinstance(event, StreamChatDeltaReceivedEvent):
            logger.debug(event.delta)
        if isinstance(event, StreamChatErrorEvent):
            logger.debug(event.exception)
        if isinstance(event, EmbeddingStartEvent):
            logger.debug(event.model_dict)
        if isinstance(event, EmbeddingEndEvent):
            logger.debug(event.chunks)
            logger.debug(event.embeddings[0][:5])  # avoid printing all embeddings
        if isinstance(event, LLMPredictStartEvent):
            logger.debug(event.template)
            logger.debug(event.template_args)
        if isinstance(event, LLMPredictEndEvent):
            logger.debug(event.output)
        if isinstance(event, LLMStructuredPredictStartEvent):
            logger.debug(event.template)
            logger.debug(event.template_args)
            logger.debug(event.output_cls)
        if isinstance(event, LLMStructuredPredictEndEvent):
            logger.debug(event.output)
        if isinstance(event, LLMCompletionStartEvent):
            logger.debug(event.model_dict)
            logger.debug(event.prompt)
            logger.debug(event.additional_kwargs)
        if isinstance(event, LLMCompletionEndEvent):
            logger.debug(event.response)
            logger.debug(event.prompt)
        if isinstance(event, LLMChatInProgressEvent):
            logger.debug(event.messages)
            logger.debug(event.response)
        if isinstance(event, LLMChatStartEvent):
            logger.debug(event.messages)
            logger.debug(event.additional_kwargs)
            logger.debug(event.model_dict)
        if isinstance(event, LLMChatEndEvent):
            logger.debug(event.messages)
            logger.debug(event.response)
        if isinstance(event, RetrievalStartEvent):
            logger.debug(event.str_or_query_bundle)
        if isinstance(event, RetrievalEndEvent):
            logger.debug(event.str_or_query_bundle)
            logger.debug(event.nodes)
        if isinstance(event, ReRankStartEvent):
            logger.debug(event.query)
            logger.debug(event.nodes)
            logger.debug(event.top_n)
            logger.debug(event.model_name)
        if isinstance(event, ReRankEndEvent):
            logger.debug(event.nodes)
        if isinstance(event, QueryStartEvent):
            logger.debug(event.query)
        if isinstance(event, QueryEndEvent):
            logger.debug(event.response)
            logger.debug(event.query)
        if isinstance(event, SpanDropEvent):
            logger.debug(event.err_str)
        if isinstance(event, SynthesizeStartEvent):
            logger.debug(event.query)
        if isinstance(event, SynthesizeEndEvent):
            logger.debug(event.response)
            logger.debug(event.query)
        if isinstance(event, GetResponseStartEvent):
            logger.debug(event.query_str)

        self.events.append(event)
        logger.debug("-----------------------")

    def _get_events_by_span(self) -> Dict[str, List[BaseEvent]]:
        events_by_span: Dict[str, List[BaseEvent]] = {}
        for event in self.events:
            if event.span_id in events_by_span:
                events_by_span[event.span_id].append(event)
            else:
                events_by_span[event.span_id] = [event]
        return events_by_span

    def _get_event_span_trees(self) -> List[Tree]:
        events_by_span = self._get_events_by_span()

        trees = []
        tree = Tree()

        for span, sorted_events in events_by_span.items():
            tree.create_node(
                tag=f"{span} (SPAN)",
                identifier=span,
                parent=None,
                data=sorted_events[0].timestamp,
            )

            for event in sorted_events:
                tree.create_node(
                    tag=f"{event.class_name()}: {event.id_}",
                    identifier=event.id_,
                    parent=event.span_id,
                    data=event.timestamp,
                )

            trees.append(tree)
            tree = Tree()
        return trees

    def print_event_span_trees(self) -> None:
        """Method for viewing trace trees."""
        trees = self._get_event_span_trees()
        for tree in trees:
            logger.debug(
                tree.show(
                    stdout=False, sorting=True, key=lambda node: node.data
                )
            )
            logger.debug("")

"""
## Spans

Spans are "operations" in LlamaIndex (typically function calls). Spans can contain more spans, and each span contains associated events.

The below code shows how to observe spans as they happen in LlamaIndex
"""
logger.info("## Spans")




class ExampleSpanHandler(BaseSpanHandler[SimpleSpan]):
    span_dict = {}

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "ExampleSpanHandler"

    def new_span(
        self,
        id_: str,
        bound_args: Any,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[SimpleSpan]:
        """Create a span."""
        if id_ not in self.span_dict:
            self.span_dict[id_] = []
        self.span_dict[id_].append(
            SimpleSpan(id_=id_, parent_id=parent_span_id)
        )

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: Any,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Logic for preparing to exit a span."""
        pass

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: Any,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> Any:
        """Logic for preparing to drop a span."""
        pass

"""
## Putting it all Together

With our span handler and event handler defined, we can attach it to a dispatcher watch events and spans come in.

It is not mandatory to have both a span handler and event handler, you could have either-or, or both.
"""
logger.info("## Putting it all Together")


root_dispatcher = get_dispatcher()

event_handler = ExampleEventHandler()
span_handler = ExampleSpanHandler()
simple_span_handler = SimpleSpanHandler()
root_dispatcher.add_span_handler(span_handler)
root_dispatcher.add_span_handler(simple_span_handler)
root_dispatcher.add_event_handler(event_handler)


# os.environ["OPENAI_API_KEY"] = "sk-..."


index = VectorStoreIndex.from_documents([Document.example()])

query_engine = index.as_query_engine()

query_engine.query("Tell me about LLMs?")

event_handler.print_event_span_trees()

simple_span_handler.print_trace_trees()

logger.info("\n\n[DONE]", bright=True)