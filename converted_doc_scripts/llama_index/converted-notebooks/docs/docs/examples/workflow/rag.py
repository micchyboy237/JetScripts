async def main():
    from jet.models.config import MODELS_CACHE_DIR
    from jet.transformers.formatters import format_json
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
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
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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
    # !wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"
    
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
                embed_model=HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR),
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
    
            await ctx.store.set("query", query)
    
            if index is None:
                logger.debug("Index is empty, load some documents before querying!")
                return None
    
            retriever = index.as_retriever(similarity_top_k=2)
            nodes = await retriever.aretrieve(query)
            logger.success(format_json(nodes))
            logger.success(format_json(nodes))
            logger.debug(f"Retrieved {len(nodes)} nodes.")
            return RetrieverEvent(nodes=nodes)
    
        @step
        async def rerank(self, ctx: Context, ev: RetrieverEvent) -> RerankEvent:
            ranker = LLMRerank(
                choice_batch_size=5, top_n=3, llm=OllamaFunctionCallingAdapter(model="llama3.2")
            )
            logger.debug(await ctx.store.get("query", default=None), flush=True)
            new_nodes = ranker.postprocess_nodes(
                ev.nodes, query_str=await ctx.store.get("query", default=None)
            )
            logger.debug(f"Reranked nodes to {len(new_nodes)}")
            return RerankEvent(nodes=new_nodes)
    
        @step
        async def synthesize(self, ctx: Context, ev: RerankEvent) -> StopEvent:
            """Return a streaming response using reranked nodes."""
            llm = OllamaFunctionCallingAdapter(model="llama3.2")
            summarizer = CompactAndRefine(llm=llm, streaming=True, verbose=True)
            query = await ctx.store.get("query", default=None)
            logger.success(format_json(query))
            logger.success(format_json(query))
    
            response = await summarizer.asynthesize(query, nodes=ev.nodes)
            logger.success(format_json(response))
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
    
    index = await w.run(dirname="data")
    logger.success(format_json(index))
    logger.success(format_json(index))
    
    result = await w.run(query="How was Llama2 trained?", index=index)
    logger.success(format_json(result))
    logger.success(format_json(result))
    async for chunk in result.async_response_gen():
        logger.debug(chunk, end="", flush=True)
    
    logger.info("\n\n[DONE]", bright=True)

if __name__ == '__main__':
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())