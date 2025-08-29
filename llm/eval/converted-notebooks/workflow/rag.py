from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from jet.llm.ollama.base import OllamaEmbedding
from jet.llm.ollama.base import Ollama
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.schema import NodeWithScore
from llama_index.core.workflow import Event
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HTTPSpanExporter,
)
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk import trace as trace_sdk
import os
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

# RAG Workflow with Reranking
#
# This notebook walks through setting up a `Workflow` to perform basic RAG with reranking.

# !pip install -U llama-index


# os.environ["OPENAI_API_KEY"] = "sk-proj-..."

# [Optional] Set up observability with Llamatrace
#
# Set up tracing to visualize each step in the workflow.

# %pip install "openinference-instrumentation-llama-index>=3.0.0" "opentelemetry-proto>=1.12.0" opentelemetry-exporter-otlp opentelemetry-sdk


PHOENIX_API_KEY = "<YOUR-PHOENIX-API-KEY>"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={PHOENIX_API_KEY}"

span_phoenix_processor = SimpleSpanProcessor(
    HTTPSpanExporter(endpoint="https://app.phoenix.arize.com/v1/traces")
)

tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(span_processor=span_phoenix_processor)

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

# !mkdir -p data
# !wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"

# Since workflows are async first, this all runs fine in a notebook. If you were running in your own code, you would want to use `asyncio.run()` to start an async event loop if one isn't already running.
#
# ```python
# async def main():
#     <async code>
#
# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())
# ```

# Designing the Workflow
#
# RAG + Reranking consists of some clearly defined steps
# 1. Indexing data, creating an index
# 2. Using that index + a query to retrieve relevant text chunks
# 3. Rerank the text retrieved text chunks using the original query
# 4. Synthesizing a final response
#
# With this in mind, we can create events and workflow steps to follow this process!
#
# The Workflow Events
#
# To handle these steps, we need to define a few events:
# 1. An event to pass retrieved nodes to the reranker
# 2. An event to pass reranked nodes to the synthesizer
#
# The other steps will use the built-in `StartEvent` and `StopEvent` events.


class RetrieverEvent(Event):
    """Result of running retrieval"""

    nodes: list[NodeWithScore]


class RerankEvent(Event):
    """Result of running reranking on retrieved nodes"""

    nodes: list[NodeWithScore]

# The Workflow Itself
#
# With our events defined, we can construct our workflow and steps.
#
# Note that the workflow automatically validates itself using type annotations, so the type annotations on our steps are very helpful!


class RAGWorkflow(Workflow):
    @step
    async def ingest(self, ctx: Context, ev: StartEvent) -> StopEvent | None:
        """Entry point to ingest a document, triggered by a StartEvent with `dirname`."""
        dirname = ev.get("dirname")
        if not dirname:
            return None

        documents = SimpleDirectoryReader(dirname).load_data()
        index = VectorStoreIndex.from_documents(
            documents=documents,
            embed_model=OllamaEmbedding(model_name="nomic-embed-text"),
        )
        return StopEvent(result=index)

    @step
    async def retrieve(
        self, ctx: Context, ev: StartEvent
    ) -> RetrieverEvent | None:
        "Entry point for RAG, triggered by a StartEvent with `query`."
        query = ev.get("query")
        index = ev.get("index")

        if not query:
            return None

        print(f"Query the database with: {query}")

        await ctx.set("query", query)

        if index is None:
            print("Index is empty, load some documents before querying!")
            return None

        retriever = index.as_retriever(similarity_top_k=2)
        nodes = await retriever.aretrieve(query)
        print(f"Retrieved {len(nodes)} nodes.")
        return RetrieverEvent(nodes=nodes)

    @step
    async def rerank(self, ctx: Context, ev: RetrieverEvent) -> RerankEvent:
        ranker = LLMRerank(
            choice_batch_size=5, top_n=3, llm=Ollama(model="llama3.1")
        )
        print(await ctx.get("query", default=None), flush=True)
        new_nodes = ranker.postprocess_nodes(
            ev.nodes, query_str=await ctx.get("query", default=None)
        )
        print(f"Reranked nodes to {len(new_nodes)}")
        return RerankEvent(nodes=new_nodes)

    @step
    async def synthesize(self, ctx: Context, ev: RerankEvent) -> StopEvent:
        """Return a streaming response using reranked nodes."""
        llm = Ollama(model="llama3.1", request_timeout=300.0,
                     context_window=4096)
        summarizer = CompactAndRefine(llm=llm, streaming=True, verbose=True)
        query = await ctx.get("query", default=None)

        response = await summarizer.asynthesize(query, nodes=ev.nodes)
        return StopEvent(result=response)

# And thats it! Let's explore the workflow we wrote a bit.
#
# - We have two entry points (the steps that accept `StartEvent`)
# - The steps themselves decide when they can run
# - The workflow context is used to store the user query
# - The nodes are passed around, and finally a streaming response is returned

# Run the Workflow!


w = RAGWorkflow()

index = await w.run(dirname="data")

result = await w.run(query="How was Llama2 trained?", index=index)
async for chunk in result.async_response_gen():
    print(chunk, end="", flush=True)

logger.info("\n\n[DONE]", bright=True)
