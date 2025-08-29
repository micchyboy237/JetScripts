async def main():
    from jet.models.config import MODELS_CACHE_DIR
    from jet.transformers.formatters import format_json
    from IPython.display import Markdown, display
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.prompts import PromptTemplate
    from llama_index.core.response_synthesizers import (
    ResponseMode,
    get_response_synthesizer,
    )
    from llama_index.core.schema import (
    MetadataMode,
    NodeWithScore,
    TextNode,
    )
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
    from typing import Union, List
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
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/workflow/citation_query_engine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    # Build RAG with in-line citations
    
    This notebook walks through implementation of RAG with in-line citations of source nodes, using Workflows.
    
    Specifically we will implement [CitationQueryEngine](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/query_engine/citation_query_engine.ipynb) which gives in-line citations in the response generated based on the nodes.
    """
    logger.info("# Build RAG with in-line citations")
    
    # !pip install -U llama-index
    
    
    # os.environ["OPENAI_API_KEY"] = "sk-..."
    
    """
    ## Download Data
    """
    logger.info("## Download Data")
    
    # !mkdir -p 'data/paul_graham/'
    # !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'
    
    """
    Since workflows are async first, this all runs fine in a notebook. If you were running in your own code, you would want to use `asyncio.run()` to start an async event loop if one isn't already running.
    
    ```python
    async def main():
        <async code>
    
    if __name__ == "__main__":
        asyncio.run(main())
    ```
    
    ## Designing the Workflow
    
    CitationQueryEngine consists of some clearly defined steps
    1. Indexing data, creating an index
    2. Using that index + a query to retrieve relevant nodes
    3. Add citations to the retrieved nodes.
    4. Synthesizing a final response
    
    With this in mind, we can create events and workflow steps to follow this process!
    
    ### The Workflow Events
    
    To handle these steps, we need to define a few events:
    1. An event to pass retrieved nodes to the create citations
    2. An event to pass citation nodes to the synthesizer
    
    The other steps will use the built-in `StartEvent` and `StopEvent` events.
    """
    logger.info("## Designing the Workflow")
    
    
    
    class RetrieverEvent(Event):
        """Result of running retrieval"""
    
        nodes: list[NodeWithScore]
    
    
    class CreateCitationsEvent(Event):
        """Add citations to the nodes."""
    
        nodes: list[NodeWithScore]
    
    """
    ## Citation Prompt Templates
    
    Here we define default `CITATION_QA_TEMPLATE`, `CITATION_REFINE_TEMPLATE`, `DEFAULT_CITATION_CHUNK_SIZE` and `DEFAULT_CITATION_CHUNK_OVERLAP`.
    """
    logger.info("## Citation Prompt Templates")
    
    
    CITATION_QA_TEMPLATE = PromptTemplate(
        "Please provide an answer based solely on the provided sources. "
        "When referencing information from a source, "
        "cite the appropriate source(s) using their corresponding numbers. "
        "Every answer should include at least one source citation. "
        "Only cite a source when you are explicitly referencing it. "
        "If none of the sources are helpful, you should indicate that. "
        "For example:\n"
        "Source 1:\n"
        "The sky is red in the evening and blue in the morning.\n"
        "Source 2:\n"
        "Water is wet when the sky is red.\n"
        "Query: When is water wet?\n"
        "Answer: Water will be wet when the sky is red [2], "
        "which occurs in the evening [1].\n"
        "Now it's your turn. Below are several numbered sources of information:"
        "\n------\n"
        "{context_str}"
        "\n------\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    
    CITATION_REFINE_TEMPLATE = PromptTemplate(
        "Please provide an answer based solely on the provided sources. "
        "When referencing information from a source, "
        "cite the appropriate source(s) using their corresponding numbers. "
        "Every answer should include at least one source citation. "
        "Only cite a source when you are explicitly referencing it. "
        "If none of the sources are helpful, you should indicate that. "
        "For example:\n"
        "Source 1:\n"
        "The sky is red in the evening and blue in the morning.\n"
        "Source 2:\n"
        "Water is wet when the sky is red.\n"
        "Query: When is water wet?\n"
        "Answer: Water will be wet when the sky is red [2], "
        "which occurs in the evening [1].\n"
        "Now it's your turn. "
        "We have provided an existing answer: {existing_answer}"
        "Below are several numbered sources of information. "
        "Use them to refine the existing answer. "
        "If the provided sources are not helpful, you will repeat the existing answer."
        "\nBegin refining!"
        "\n------\n"
        "{context_msg}"
        "\n------\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    
    DEFAULT_CITATION_CHUNK_SIZE = 512
    DEFAULT_CITATION_CHUNK_OVERLAP = 20
    
    """
    ### The Workflow Itself
    
    With our events defined, we can construct our workflow and steps. 
    
    Note that the workflow automatically validates itself using type annotations, so the type annotations on our steps are very helpful!
    """
    logger.info("### The Workflow Itself")
    
    
    
    
    
    
    
    class CitationQueryEngineWorkflow(Workflow):
        @step
        async def retrieve(
            self, ctx: Context, ev: StartEvent
        ) -> Union[RetrieverEvent, None]:
            "Entry point for RAG, triggered by a StartEvent with `query`."
            query = ev.get("query")
            if not query:
                return None
    
            logger.debug(f"Query the database with: {query}")
    
            await ctx.store.set("query", query)
    
            if ev.index is None:
                logger.debug("Index is empty, load some documents before querying!")
                return None
    
            retriever = ev.index.as_retriever(similarity_top_k=2)
            nodes = retriever.retrieve(query)
            logger.debug(f"Retrieved {len(nodes)} nodes.")
            return RetrieverEvent(nodes=nodes)
    
        @step
        async def create_citation_nodes(
            self, ev: RetrieverEvent
        ) -> CreateCitationsEvent:
            """
            Modify retrieved nodes to create granular sources for citations.
    
            Takes a list of NodeWithScore objects and splits their content
            into smaller chunks, creating new NodeWithScore objects for each chunk.
            Each new node is labeled as a numbered source, allowing for more precise
            citation in query results.
    
            Args:
                nodes (List[NodeWithScore]): A list of NodeWithScore objects to be processed.
    
            Returns:
                List[NodeWithScore]: A new list of NodeWithScore objects, where each object
                represents a smaller chunk of the original nodes, labeled as a source.
            """
            nodes = ev.nodes
    
            new_nodes: List[NodeWithScore] = []
    
            text_splitter = SentenceSplitter(
                chunk_size=DEFAULT_CITATION_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CITATION_CHUNK_OVERLAP,
            )
    
            for node in nodes:
                text_chunks = text_splitter.split_text(
                    node.node.get_content(metadata_mode=MetadataMode.NONE)
                )
    
                for text_chunk in text_chunks:
                    text = f"Source {len(new_nodes)+1}:\n{text_chunk}\n"
    
                    new_node = NodeWithScore(
                        node=TextNode.parse_obj(node.node), score=node.score
                    )
                    new_node.node.text = text
                    new_nodes.append(new_node)
            return CreateCitationsEvent(nodes=new_nodes)
    
        @step
        async def synthesize(
            self, ctx: Context, ev: CreateCitationsEvent
        ) -> StopEvent:
            """Return a streaming response using the retrieved nodes."""
            llm = OllamaFunctionCallingAdapter(model="llama3.2")
            query = await ctx.store.get("query", default=None)
            logger.success(format_json(query))
    
            synthesizer = get_response_synthesizer(
                llm=llm,
                text_qa_template=CITATION_QA_TEMPLATE,
                refine_template=CITATION_REFINE_TEMPLATE,
                response_mode=ResponseMode.COMPACT,
                use_async=True,
            )
    
            response = await synthesizer.asynthesize(query, nodes=ev.nodes)
            logger.success(format_json(response))
            return StopEvent(result=response)
    
    """
    And thats it! Let's explore the workflow we wrote a bit.
    
    - We have an entry point (the step that accept `StartEvent`)
    - The workflow context is used to store the user query
    - The nodes are retrieved, citations are created, and finally a response is returned
    
    ## Create Index
    """
    logger.info("## Create Index")
    
    documents = SimpleDirectoryReader("data/paul_graham").load_data()
    index = VectorStoreIndex.from_documents(
        documents=documents,
        embed_model=HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR),
    )
    
    """
    ### Run the Workflow!
    """
    logger.info("### Run the Workflow!")
    
    w = CitationQueryEngineWorkflow()
    
    result = await w.run(query="What information do you have", index=index)
    logger.success(format_json(result))
    
    
    display(Markdown(f"{result}"))
    
    """
    ## Check the citations.
    """
    logger.info("## Check the citations.")
    
    logger.debug(result.source_nodes[0].node.get_text())
    
    logger.debug(result.source_nodes[1].node.get_text())
    
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