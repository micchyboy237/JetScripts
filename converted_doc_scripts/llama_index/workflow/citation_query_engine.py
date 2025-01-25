import asyncio
from jet.llm.ollama.constants import OLLAMA_BASE_EMBED_URL
from jet.transformers.formatters import format_json
from llama_index.core.prompts import PromptTemplate
from IPython.display import Markdown, display
from llama_index.core.node_parser import SentenceSplitter
from typing import Union, List
from llama_index.core.response_synthesizers import (
    ResponseMode,
    get_response_synthesizer,
)
from llama_index.core.schema import (
    MetadataMode,
    NodeWithScore,
    TextNode,
)
from jet.llm.ollama.base import OllamaEmbedding
from jet.llm.ollama.base import Ollama
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.workflow import Event
import os
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/workflow/citation_query_engine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Build RAG with in-line citations
#
# This notebook walks through implementation of RAG with in-line citations of source nodes, using Workflows.
#
# Specifically we will implement [CitationQueryEngine](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/query_engine/citation_query_engine.ipynb) which gives in-line citations in the response generated based on the nodes.

# !pip install -U llama-index


# os.environ["OPENAI_API_KEY"] = "sk-..."

# Download Data

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

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
# CitationQueryEngine consists of some clearly defined steps
# 1. Indexing data, creating an index
# 2. Using that index + a query to retrieve relevant nodes
# 3. Add citations to the retrieved nodes.
# 4. Synthesizing a final response
#
# With this in mind, we can create events and workflow steps to follow this process!
#
# The Workflow Events
#
# To handle these steps, we need to define a few events:
# 1. An event to pass retrieved nodes to the create citations
# 2. An event to pass citation nodes to the synthesizer
#
# The other steps will use the built-in `StartEvent` and `StopEvent` events.


class RetrieverEvent(Event):
    """Result of running retrieval"""

    nodes: list[NodeWithScore]


class CreateCitationsEvent(Event):
    """Add citations to the nodes."""

    nodes: list[NodeWithScore]

# Citation Prompt Templates
#
# Here we define default `CITATION_QA_TEMPLATE`, `CITATION_REFINE_TEMPLATE`, `DEFAULT_CITATION_CHUNK_SIZE` and `DEFAULT_CITATION_CHUNK_OVERLAP`.


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

# The Workflow Itself
#
# With our events defined, we can construct our workflow and steps.
#
# Note that the workflow automatically validates itself using type annotations, so the type annotations on our steps are very helpful!


class CitationQueryEngineWorkflow(Workflow):
    @step
    async def retrieve(
        self, ctx: Context, ev: StartEvent
    ) -> Union[RetrieverEvent, None]:
        "Entry point for RAG, triggered by a StartEvent with `query`."
        query = ev.get("query")
        if not query:
            return None

        print(f"Query the database with: {query}")

        await ctx.set("query", query)

        if ev.index is None:
            print("Index is empty, load some documents before querying!")
            return None

        retriever = ev.index.as_retriever(similarity_top_k=2)
        nodes = retriever.retrieve(query)
        print(f"Retrieved {len(nodes)} nodes.")
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
        llm = Ollama(model="llama3.1", request_timeout=300.0,
                     context_window=4096)
        query = await ctx.get("query", default=None)

        synthesizer = get_response_synthesizer(
            llm=llm,
            text_qa_template=CITATION_QA_TEMPLATE,
            refine_template=CITATION_REFINE_TEMPLATE,
            response_mode=ResponseMode.COMPACT,
            use_async=True,
        )

        response = await synthesizer.asynthesize(query, nodes=ev.nodes)
        return StopEvent(result=response)

# And thats it! Let's explore the workflow we wrote a bit.
#
# - We have an entry point (the step that accept `StartEvent`)
# - The workflow context is used to store the user query
# - The nodes are retrieved, citations are created, and finally a response is returned

# Create Index


documents = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()
results_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/results/citations.json"

index = VectorStoreIndex.from_documents(
    documents=documents,
    embed_model=OllamaEmbedding(
        model_name="nomic-embed-text", base_url=OLLAMA_BASE_EMBED_URL),
)


async def run_workflow():
    # Run the Workflow!

    w = CitationQueryEngineWorkflow()

    query = "What information do you have"
    result = await w.run(query=query, index=index)

    logger.newline()
    logger.info("Query:")
    logger.debug(query)
    logger.success(format_json(result))

    # Check the citations.
    logger.newline()
    logger.info("Citation 1:")
    logger.success(result.source_nodes[0].node.get_text())
    logger.info("Citation 2:")
    logger.success(result.source_nodes[1].node.get_text())

    from jet.file import save_file
    save_file(result, results_path)

asyncio.run(run_workflow())

logger.info("\n\n[DONE]", bright=True)
