from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/workflow/router_query_engine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Router Query Engine
# 
# `RouterQueryEngine` chooses the most appropriate query engine from multiple options to process a given query.
# 
# This notebook walks through implementation of Router Query Engine, using workflows.
# 
# Specifically we will implement [RouterQueryEngine](https://docs.llamaindex.ai/en/stable/examples/query_engine/RouterQueryEngine/).

# !pip install -U llama-index

import os

# os.environ["OPENAI_API_KEY"] = "sk-.."

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

## Define Events

from llama_index.core.workflow import Event
from llama_index.core.base.base_selector import SelectorResult
from typing import Dict, List, Any
from llama_index.core.base.response.schema import RESPONSE_TYPE


class QueryEngineSelectionEvent(Event):
    """Result of selecting the query engine tools."""

    selected_query_engines: SelectorResult


class SynthesizeEvent(Event):
    """Event for synthesizing the response from different query engines."""

    result: List[RESPONSE_TYPE]
    selected_query_engines: SelectorResult

## The Workflow
# 
# `selector:`
# 
# 1. It takes a StartEvent as input and returns a QueryEngineSelectionEvent.
# 2. The `LLMSingleSelector`/ `PydanticSingleSelector`/ `PydanticMultiSelector` will select one/ multiple query engine tools.
# 
# `generate_responses:`
# 
# This function uses the selected query engines to generate responses and returns SynthesizeEvent.
# 
# `synthesize_responses:`
# 
# This function combines the generated responses and synthesizes the final response if multiple query engines are selected otherwise returns the single generated response.
# 
# 
# The steps will use the built-in `StartEvent` and `StopEvent` events.
# 
# With our events defined, we can construct our workflow and steps.

from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)

from llama_index.llms.ollama import Ollama
from llama_index.core.selectors.utils import get_selector_from_llm
from llama_index.core.base.response.schema import (
    PydanticResponse,
    Response,
    AsyncStreamingResponse,
)
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.schema import QueryBundle
from llama_index.core import Settings

from IPython.display import Markdown, display
import asyncio


class RouterQueryEngineWorkflow(Workflow):
    @step
    async def selector(
        self, ctx: Context, ev: StartEvent
    ) -> QueryEngineSelectionEvent:
        """
        Selects a single/ multiple query engines based on the query.
        """

        await ctx.set("query", ev.get("query"))
        await ctx.set("llm", ev.get("llm"))
        await ctx.set("query_engine_tools", ev.get("query_engine_tools"))
        await ctx.set("summarizer", ev.get("summarizer"))

        llm = Settings.llm
        select_multiple_query_engines = ev.get("select_multi")
        query = ev.get("query")
        query_engine_tools = ev.get("query_engine_tools")

        selector = get_selector_from_llm(
            llm, is_multi=select_multiple_query_engines
        )

        query_engines_metadata = [
            query_engine.metadata for query_engine in query_engine_tools
        ]

        selected_query_engines = await selector.aselect(
            query_engines_metadata, query
        )

        return QueryEngineSelectionEvent(
            selected_query_engines=selected_query_engines
        )

    @step
    async def generate_responses(
        self, ctx: Context, ev: QueryEngineSelectionEvent
    ) -> SynthesizeEvent:
        """Generate the responses from the selected query engines."""

        query = await ctx.get("query", default=None)
        selected_query_engines = ev.selected_query_engines
        query_engine_tools = await ctx.get("query_engine_tools")

        query_engines = [engine.query_engine for engine in query_engine_tools]

        print(
            f"number of selected query engines: {len(selected_query_engines.selections)}"
        )

        if len(selected_query_engines.selections) > 1:
            tasks = []
            for selected_query_engine in selected_query_engines.selections:
                print(
                    f"Selected query engine: {selected_query_engine.index}: {selected_query_engine.reason}"
                )
                query_engine = query_engines[selected_query_engine.index]
                tasks.append(query_engine.aquery(query))

            response_generated = await asyncio.gather(*tasks)

        else:
            query_engine = query_engines[
                selected_query_engines.selections[0].index
            ]

            print(
                f"Selected query engine: {selected_query_engines.ind}: {selected_query_engines.reason}"
            )

            response_generated = [query_engine.query(query)]

        return SynthesizeEvent(
            result=response_generated,
            selected_query_engines=selected_query_engines,
        )

    async def acombine_responses(
        self,
        summarizer: TreeSummarize,
        responses: List[RESPONSE_TYPE],
        query_bundle: QueryBundle,
    ) -> RESPONSE_TYPE:
        """Async combine multiple response from sub-engines."""

        print("Combining responses from multiple query engines.")

        response_strs = []
        source_nodes = []
        for response in responses:
            if isinstance(
                response, (AsyncStreamingResponse, PydanticResponse)
            ):
                response_obj = response.get_response()
            else:
                response_obj = response
            source_nodes.extend(response_obj.source_nodes)
            response_strs.append(str(response))

        summary = summarizer.get_response(
            query_bundle.query_str, response_strs
        )

        if isinstance(summary, str):
            return Response(response=summary, source_nodes=source_nodes)
        elif isinstance(summary, BaseModel):
            return PydanticResponse(
                response=summary, source_nodes=source_nodes
            )
        else:
            return AsyncStreamingResponse(
                response_gen=summary, source_nodes=source_nodes
            )

    @step
    async def synthesize_responses(
        self, ctx: Context, ev: SynthesizeEvent
    ) -> StopEvent:
        """Synthesizes the responses from the generated responses."""

        response_generated = ev.result
        query = await ctx.get("query", default=None)
        summarizer = await ctx.get("summarizer")
        selected_query_engines = ev.selected_query_engines

        if len(response_generated) > 1:
            response = await self.acombine_responses(
                summarizer, response_generated, QueryBundle(query_str=query)
            )
        else:
            response = response_generated[0]

        response.metadata = response.metadata or {}
        response.metadata["selector_result"] = selected_query_engines

        return StopEvent(result=response)

## Define LLM

llm = Ollama(model="llama3.1", request_timeout=300.0, context_window=4096)
Settings.llm = llm

## Define Summarizer

from llama_index.core.prompts.default_prompt_selectors import (
    DEFAULT_TREE_SUMMARIZE_PROMPT_SEL,
)

summarizer = TreeSummarize(
    llm=llm,
    summary_template=DEFAULT_TREE_SUMMARIZE_PROMPT_SEL,
)

## Download Data

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

## Load Data

from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/eval/converted-notebooks/retrievers/data/jet-resume").load_data()

## Create Nodes

nodes = Settings.node_parser.get_nodes_from_documents(documents)

## Create Indices
# 
# We will create three indices SummaryIndex, VectorStoreIndex and SimpleKeywordTableIndex.

from llama_index.core import (
    VectorStoreIndex,
    SummaryIndex,
    SimpleKeywordTableIndex,
)

summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes)
keyword_index = SimpleKeywordTableIndex(nodes)

## Create Query Engine Tools

from llama_index.core.tools import QueryEngineTool

list_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
vector_query_engine = vector_index.as_query_engine()
keyword_query_engine = keyword_index.as_query_engine()

list_tool = QueryEngineTool.from_defaults(
    query_engine=list_query_engine,
    description=(
        "Useful for summarization questions related to Paul Graham eassy on"
        " What I Worked On."
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Useful for retrieving specific context from Paul Graham essay on What"
        " I Worked On."
    ),
)

keyword_tool = QueryEngineTool.from_defaults(
    query_engine=keyword_query_engine,
    description=(
        "Useful for retrieving specific context using keywords from Paul"
        " Graham essay on What I Worked On."
    ),
)

query_engine_tools = [list_tool, vector_tool, keyword_tool]

## Run the Workflow!

import nest_asyncio

nest_asyncio.apply()

w = RouterQueryEngineWorkflow(timeout=200)

### Querying

#### Summarization Query

query = "Provide the summary of the document?"

result = await w.run(
    query=query,
    llm=llm,
    query_engine_tools=query_engine_tools,
    summarizer=summarizer,
    select_multi=True,  # You can change it to default it to select only one query engine.
)

display(
    Markdown("> Question: {}".format(query)),
    Markdown("Answer: {}".format(result)),
)

#### Pointed Context Query

query = "What did the author do growing up?"

result = await w.run(
    query=query,
    llm=llm,
    query_engine_tools=query_engine_tools,
    summarizer=summarizer,
    select_multi=False,  # You can change it to select multiple query engines.
)

display(
    Markdown("> Question: {}".format(query)),
    Markdown("Answer: {}".format(result)),
)

query = "What were noteable events and people from the authors time at Interleaf and YC?"

result = await w.run(
    query=query,
    llm=llm,
    query_engine_tools=query_engine_tools,
    summarizer=summarizer,
    select_multi=True,  # Since query should use two query engine tools, we enabled it.
)

display(
    Markdown("> Question: {}".format(query)),
    Markdown("Answer: {}".format(result)),
)

logger.info("\n\n[DONE]", bright=True)