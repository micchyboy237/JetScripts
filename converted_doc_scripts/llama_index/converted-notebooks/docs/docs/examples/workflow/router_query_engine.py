import asyncio
from jet.transformers.formatters import format_json
from IPython.display import Markdown, display
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import (
VectorStoreIndex,
SummaryIndex,
SimpleKeywordTableIndex,
)
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core.base.base_selector import SelectorResult
from llama_index.core.base.response.schema import (
PydanticResponse,
Response,
AsyncStreamingResponse,
)
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.prompts.default_prompt_selectors import (
DEFAULT_TREE_SUMMARIZE_PROMPT_SEL,
)
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.schema import QueryBundle
from llama_index.core.selectors.utils import get_selector_from_llm
from llama_index.core.settings import Settings
from llama_index.core.tools import QueryEngineTool
from llama_index.core.workflow import (
Context,
Workflow,
StartEvent,
StopEvent,
step,
)
from llama_index.core.workflow import Event
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import Dict, List, Any
import asyncio
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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/workflow/router_query_engine.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Router Query Engine

`RouterQueryEngine` chooses the most appropriate query engine from multiple options to process a given query.

This notebook walks through implementation of Router Query Engine, using workflows.

Specifically we will implement [RouterQueryEngine](https://docs.llamaindex.ai/en/stable/examples/query_engine/RouterQueryEngine/).
"""
logger.info("# Router Query Engine")

# !pip install -U llama-index


# os.environ["OPENAI_API_KEY"] = "sk-.."

"""
Since workflows are async first, this all runs fine in a notebook. If you were running in your own code, you would want to use `asyncio.run()` to start an async event loop if one isn't already running.

```python
async def main():
    <async code>

if __name__ == "__main__":
    asyncio.run(main())
```

## Define Events
"""
logger.info("## Define Events")



class QueryEngineSelectionEvent(Event):
    """Result of selecting the query engine tools."""

    selected_query_engines: SelectorResult


class SynthesizeEvent(Event):
    """Event for synthesizing the response from different query engines."""

    result: List[RESPONSE_TYPE]
    selected_query_engines: SelectorResult

"""
## The Workflow

`selector:`

1. It takes a StartEvent as input and returns a QueryEngineSelectionEvent.
2. The `LLMSingleSelector`/ `PydanticSingleSelector`/ `PydanticMultiSelector` will select one/ multiple query engine tools.

`generate_responses:`

This function uses the selected query engines to generate responses and returns SynthesizeEvent.

`synthesize_responses:`

This function combines the generated responses and synthesizes the final response if multiple query engines are selected otherwise returns the single generated response.


The steps will use the built-in `StartEvent` and `StopEvent` events.

With our events defined, we can construct our workflow and steps.
"""
logger.info("## The Workflow")





class RouterQueryEngineWorkflow(Workflow):
    @step
    async def selector(
        self, ctx: Context, ev: StartEvent
    ) -> QueryEngineSelectionEvent:
        """
        Selects a single/ multiple query engines based on the query.
        """

        async def run_async_code_b2b0f167():
            await ctx.store.set("query", ev.get("query"))
            return 
         = asyncio.run(run_async_code_b2b0f167())
        logger.success(format_json())
        async def run_async_code_252c87dd():
            await ctx.store.set("llm", ev.get("llm"))
            return 
         = asyncio.run(run_async_code_252c87dd())
        logger.success(format_json())
        async def run_async_code_ce55df76():
            await ctx.store.set("query_engine_tools", ev.get("query_engine_tools"))
            return 
         = asyncio.run(run_async_code_ce55df76())
        logger.success(format_json())
        async def run_async_code_edf08be2():
            await ctx.store.set("summarizer", ev.get("summarizer"))
            return 
         = asyncio.run(run_async_code_edf08be2())
        logger.success(format_json())

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

        async def async_func_51():
            selected_query_engines = await selector.aselect(
                query_engines_metadata, query
            )
            return selected_query_engines
        selected_query_engines = asyncio.run(async_func_51())
        logger.success(format_json(selected_query_engines))

        return QueryEngineSelectionEvent(
            selected_query_engines=selected_query_engines
        )

    @step
    async def generate_responses(
        self, ctx: Context, ev: QueryEngineSelectionEvent
    ) -> SynthesizeEvent:
        """Generate the responses from the selected query engines."""

        async def run_async_code_a66b93bf():
            async def run_async_code_7e418399():
                query = await ctx.store.get("query", default=None)
                return query
            query = asyncio.run(run_async_code_7e418399())
            logger.success(format_json(query))
            return query
        query = asyncio.run(run_async_code_a66b93bf())
        logger.success(format_json(query))
        selected_query_engines = ev.selected_query_engines
        async def run_async_code_72c3094c():
            async def run_async_code_1928db06():
                query_engine_tools = await ctx.store.get("query_engine_tools")
                return query_engine_tools
            query_engine_tools = asyncio.run(run_async_code_1928db06())
            logger.success(format_json(query_engine_tools))
            return query_engine_tools
        query_engine_tools = asyncio.run(run_async_code_72c3094c())
        logger.success(format_json(query_engine_tools))

        query_engines = [engine.query_engine for engine in query_engine_tools]

        logger.debug(
            f"number of selected query engines: {len(selected_query_engines.selections)}"
        )

        if len(selected_query_engines.selections) > 1:
            tasks = []
            for selected_query_engine in selected_query_engines.selections:
                logger.debug(
                    f"Selected query engine: {selected_query_engine.index}: {selected_query_engine.reason}"
                )
                query_engine = query_engines[selected_query_engine.index]
                tasks.append(query_engine.aquery(query))

            async def run_async_code_5d022cee():
                async def run_async_code_37533d71():
                    response_generated = await asyncio.gather(*tasks)
                    return response_generated
                response_generated = asyncio.run(run_async_code_37533d71())
                logger.success(format_json(response_generated))
                return response_generated
            response_generated = asyncio.run(run_async_code_5d022cee())
            logger.success(format_json(response_generated))

        else:
            query_engine = query_engines[
                selected_query_engines.selections[0].index
            ]

            logger.debug(
                f"Selected query engine: {selected_query_engines.ind}: {selected_query_engines.reason}"
            )

            async def run_async_code_2e79f6a6():
                response_generated = [query_engine.query(query)]
                return 
             = asyncio.run(run_async_code_2e79f6a6())
            logger.success(format_json())

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

        logger.debug("Combining responses from multiple query engines.")

        response_strs = []
        source_nodes = []
        for response in responses:
            if isinstance(
                response, (AsyncStreamingResponse, PydanticResponse)
            ):
                async def run_async_code_4da252cf():
                    async def run_async_code_3b739492():
                        response_obj = response.get_response()
                        return response_obj
                    response_obj = asyncio.run(run_async_code_3b739492())
                    logger.success(format_json(response_obj))
                    return response_obj
                response_obj = asyncio.run(run_async_code_4da252cf())
                logger.success(format_json(response_obj))
            else:
                response_obj = response
            source_nodes.extend(response_obj.source_nodes)
            response_strs.append(str(response))

        async def async_func_124():
            summary = summarizer.get_response(
                query_bundle.query_str, response_strs
            )
            return summary
        summary = asyncio.run(async_func_124())
        logger.success(format_json(summary))

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
        async def run_async_code_a66b93bf():
            async def run_async_code_7e418399():
                query = await ctx.store.get("query", default=None)
                return query
            query = asyncio.run(run_async_code_7e418399())
            logger.success(format_json(query))
            return query
        query = asyncio.run(run_async_code_a66b93bf())
        logger.success(format_json(query))
        async def run_async_code_a45f9aa6():
            async def run_async_code_968c3653():
                summarizer = await ctx.store.get("summarizer")
                return summarizer
            summarizer = asyncio.run(run_async_code_968c3653())
            logger.success(format_json(summarizer))
            return summarizer
        summarizer = asyncio.run(run_async_code_a45f9aa6())
        logger.success(format_json(summarizer))
        selected_query_engines = ev.selected_query_engines

        if len(response_generated) > 1:
            async def async_func_151():
                response = await self.acombine_responses(
                    summarizer, response_generated, QueryBundle(query_str=query)
                )
                return response
            response = asyncio.run(async_func_151())
            logger.success(format_json(response))
        else:
            response = response_generated[0]

        response.metadata = response.metadata or {}
        response.metadata["selector_result"] = selected_query_engines

        return StopEvent(result=response)

"""
## Define LLM
"""
logger.info("## Define LLM")

llm = MLXLlamaIndexLLMAdapter(model="qwen3-1.7b-4bit-mini")
Settings.llm = llm

"""
## Define Summarizer
"""
logger.info("## Define Summarizer")


summarizer = TreeSummarize(
    llm=llm,
    summary_template=DEFAULT_TREE_SUMMARIZE_PROMPT_SEL,
)

"""
## Download Data
"""
logger.info("## Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
## Load Data
"""
logger.info("## Load Data")


documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data").load_data()

"""
## Create Nodes
"""
logger.info("## Create Nodes")

nodes = Settings.node_parser.get_nodes_from_documents(documents)

"""
## Create Indices

We will create three indices SummaryIndex, VectorStoreIndex and SimpleKeywordTableIndex.
"""
logger.info("## Create Indices")


summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes)
keyword_index = SimpleKeywordTableIndex(nodes)

"""
## Create Query Engine Tools
"""
logger.info("## Create Query Engine Tools")


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

"""
## Run the Workflow!
"""
logger.info("## Run the Workflow!")

# import nest_asyncio

# nest_asyncio.apply()

w = RouterQueryEngineWorkflow(timeout=200)

"""
### Querying

#### Summarization Query
"""
logger.info("### Querying")

query = "Provide the summary of the document?"

async def async_func_2():
    result = await w.run(
        query=query,
        llm=llm,
        query_engine_tools=query_engine_tools,
        summarizer=summarizer,
        select_multi=True,  # You can change it to default it to select only one query engine.
    )
    return result
result = asyncio.run(async_func_2())
logger.success(format_json(result))

display(
    Markdown("> Question: {}".format(query)),
    Markdown("Answer: {}".format(result)),
)

"""
#### Pointed Context Query
"""
logger.info("#### Pointed Context Query")

query = "What did the author do growing up?"

async def async_func_2():
    result = await w.run(
        query=query,
        llm=llm,
        query_engine_tools=query_engine_tools,
        summarizer=summarizer,
        select_multi=False,  # You can change it to select multiple query engines.
    )
    return result
result = asyncio.run(async_func_2())
logger.success(format_json(result))

display(
    Markdown("> Question: {}".format(query)),
    Markdown("Answer: {}".format(result)),
)

query = "What were noteable events and people from the authors time at Interleaf and YC?"

async def async_func_17():
    result = await w.run(
        query=query,
        llm=llm,
        query_engine_tools=query_engine_tools,
        summarizer=summarizer,
        select_multi=True,  # Since query should use two query engine tools, we enabled it.
    )
    return result
result = asyncio.run(async_func_17())
logger.success(format_json(result))

display(
    Markdown("> Question: {}".format(query)),
    Markdown("Answer: {}".format(result)),
)

logger.info("\n\n[DONE]", bright=True)