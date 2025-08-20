import asyncio
from jet.transformers.formatters import format_json
from jet.llm.mlx.base import MLX
from jet.llm.mlx.base import MLXEmbedding
from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.schema import NodeWithScore
from llama_index.core.workflow import (
Context,
Workflow,
StartEvent,
StopEvent,
step,
)
from llama_index.core.workflow import Event
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
OTLPSpanExporter as HTTPSpanExporter,
)
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
import asyncio
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

"""
# RAG Workflow with Reranking

This notebook walks through setting up a `Workflow` to perform basic RAG with reranking.
"""
logger.info("# RAG Workflow with Reranking")

# !pip install -U llama-index


# os.environ["OPENAI_API_KEY"] = "sk-proj-..."

"""
### [Optional] Set up observability with Llamatrace

Set up tracing to visualize each step in the workflow.
"""
logger.info("### [Optional] Set up observability with Llamatrace")

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
# !wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O f"{GENERATED_DIR}/llama2.pdf"

"""
Since workflows are async first, this all runs fine in a notebook. If you were running in your own code, you would want to use `asyncio.run()` to start an async event loop if one isn't already running.

```python
async def main():
    <async code>

if __name__ == "__main__":
    asyncio.run(main())
```

## Designing the Workflow

RAG + Reranking consists of some clearly defined steps
1. Indexing data, creating an index
2. Using that index + a query to retrieve relevant text chunks
3. Rerank the text retrieved text chunks using the original query
4. Synthesizing a final response

With this in mind, we can create events and workflow steps to follow this process!

### The Workflow Events

To handle these steps, we need to define a few events:
1. An event to pass retrieved nodes to the reranker
2. An event to pass reranked nodes to the synthesizer

The other steps will use the built-in `StartEvent` and `StopEvent` events.
"""
logger.info("## Designing the Workflow")



class RetrieverEvent(Event):
    """Result of running retrieval"""

    nodes: list[NodeWithScore]


class RerankEvent(Event):
    """Result of running reranking on retrieved nodes"""

    nodes: list[NodeWithScore]

"""
### The Workflow Itself

With our events defined, we can construct our workflow and steps. 

Note that the workflow automatically validates itself using type annotations, so the type annotations on our steps are very helpful!
"""
logger.info("### The Workflow Itself")




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
            embed_model=MLXEmbedding(model_name="mxbai-embed-large"),
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

        logger.debug(f"Query the database with: {query}")

        async def run_async_code_810a43cd():
            await ctx.store.set("query", query)
            return 
         = asyncio.run(run_async_code_810a43cd())
        logger.success(format_json())

        if index is None:
            logger.debug("Index is empty, load some documents before querying!")
            return None

        retriever = index.as_retriever(similarity_top_k=2)
        async def run_async_code_d5d11204():
            async def run_async_code_a714e101():
                nodes = await retriever.aretrieve(query)
                return nodes
            nodes = asyncio.run(run_async_code_a714e101())
            logger.success(format_json(nodes))
            return nodes
        nodes = asyncio.run(run_async_code_d5d11204())
        logger.success(format_json(nodes))
        logger.debug(f"Retrieved {len(nodes)} nodes.")
        return RetrieverEvent(nodes=nodes)

    @step
    async def rerank(self, ctx: Context, ev: RetrieverEvent) -> RerankEvent:
        ranker = LLMRerank(
            choice_batch_size=5, top_n=3, llm=MLX(model="qwen3-1.7b-4bit-mini")
        )
        async def run_async_code_2e7c416d():
            logger.debug(await ctx.store.get("query", default=None), flush=True)
            return 
         = asyncio.run(run_async_code_2e7c416d())
        logger.success(format_json())
        new_nodes = ranker.postprocess_nodes(
            async def run_async_code_4e85214a():
                ev.nodes, query_str=await ctx.store.get("query", default=None)
                return 
             = asyncio.run(run_async_code_4e85214a())
            logger.success(format_json())
        )
        logger.debug(f"Reranked nodes to {len(new_nodes)}")
        return RerankEvent(nodes=new_nodes)

    @step
    async def synthesize(self, ctx: Context, ev: RerankEvent) -> StopEvent:
        """Return a streaming response using reranked nodes."""
        llm = MLX(model="qwen3-1.7b-4bit-mini")
        summarizer = CompactAndRefine(llm=llm, streaming=True, verbose=True)
        async def run_async_code_a66b93bf():
            async def run_async_code_7e418399():
                query = await ctx.store.get("query", default=None)
                return query
            query = asyncio.run(run_async_code_7e418399())
            logger.success(format_json(query))
            return query
        query = asyncio.run(run_async_code_a66b93bf())
        logger.success(format_json(query))

        async def run_async_code_206ec8ef():
            async def run_async_code_2abdfcbc():
                response = await summarizer.asynthesize(query, nodes=ev.nodes)
                return response
            response = asyncio.run(run_async_code_2abdfcbc())
            logger.success(format_json(response))
            return response
        response = asyncio.run(run_async_code_206ec8ef())
        logger.success(format_json(response))
        return StopEvent(result=response)

"""
And thats it! Let's explore the workflow we wrote a bit.

- We have two entry points (the steps that accept `StartEvent`)
- The steps themselves decide when they can run
- The workflow context is used to store the user query
- The nodes are passed around, and finally a streaming response is returned

### Run the Workflow!
"""
logger.info("### Run the Workflow!")

w = RAGWorkflow()

async def run_async_code_0bc3fa3a():
    async def run_async_code_0beba585():
        index = await w.run(dirname="data")
        return index
    index = asyncio.run(run_async_code_0beba585())
    logger.success(format_json(index))
    return index
index = asyncio.run(run_async_code_0bc3fa3a())
logger.success(format_json(index))

async def run_async_code_72eb3343():
    async def run_async_code_e8aa5433():
        result = await w.run(query="How was Llama2 trained?", index=index)
        return result
    result = asyncio.run(run_async_code_e8aa5433())
    logger.success(format_json(result))
    return result
result = asyncio.run(run_async_code_72eb3343())
logger.success(format_json(result))
async for chunk in result.async_response_gen():
    logger.debug(chunk, end="", flush=True)

logger.info("\n\n[DONE]", bright=True)