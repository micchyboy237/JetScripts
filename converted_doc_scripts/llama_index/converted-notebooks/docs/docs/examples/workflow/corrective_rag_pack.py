import asyncio
from jet.transformers.formatters import format_json
from IPython.display import Markdown, display
from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import (
VectorStoreIndex,
Document,
PromptTemplate,
SummaryIndex,
)
from llama_index.core import SimpleDirectoryReader
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.settings import Settings
from llama_index.core.workflow import (
Workflow,
step,
Context,
StartEvent,
StopEvent,
)
from llama_index.core.workflow import Event
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.tools.tavily_research.base import TavilyToolSpec
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
# Corrective RAG Workflow

This notebook shows how to implement corrective RAG using Llamaindex workflows based on [this paper](https://arxiv.org/abs/2401.15884)

A brief understanding of the paper:


Corrective Retrieval Augmented Generation (CRAG) is a method designed to enhance the robustness of language model generation by evaluating and augmenting the relevance of retrieved documents through an evaluator and large-scale web searches, ensuring more accurate and reliable information is used in generation.

# We use `GPT-4` as a relevancy evaluator and `Tavily AI` for web searches. So, we recommend getting `OPENAI_API_KEY` and `tavily_ai_api_key` before proceeding further.
"""
logger.info("# Corrective RAG Workflow")

# import nest_asyncio

# nest_asyncio.apply()

# %pip install -U llama-index llama-index-tools-tavily-research


# os.environ["OPENAI_API_KEY"] = "sk-proj-..."
tavily_ai_api_key = os.environ["TAVILY_API_KEY"]

# !mkdir -p 'data/'
# !wget 'https://arxiv.org/pdf/2307.09288.pdf' -O 'data/llama2.pdf'

"""
Since workflows are async first, this all runs fine in a notebook. If you were running in your own code, you would want to use `asyncio.run()` to start an async event loop if one isn't already running.

```python
async def main():
    <async code>

if __name__ == "__main__":
    asyncio.run(main())
```

## Designing the Workflow

Corrective RAG consists of the following steps:
1. Ingestion of data — Loads the data into an index and setting up Tavily AI. The ingestion step will be run by itself, taking in a start event and returning a stop event.
2. Retrieval - Retrives the most relevant nodes based on the query.
3. Relevance evaluation - Uses an LLM to determine whether the retrieved nodes are relevant to the query given the content of the nodes.
4. Relevance extraction - Extracts the nodes which the LLM determined to be relevant.
5. Query transformation and Tavily search - If a node is irrelevant, then uses an LLM to transform the query to tailor towards a web search. Uses Tavily to search the web for a relevant answer based on the query.
6. Response generation - Builds a summary index given the text from the relevant nodes and the Tavily search and uses this index to get a result given the original query.

The following events are needed:
1. `PrepEvent` - Event signifying that the index and other objects are prepared.
2. `RetrieveEvent` - Event containing information about the retrieved nodes.
3. `RelevanceEvalEvent` - Event containing a list of the results of the relevance evaluation.
4. `TextExtractEvent` - Event containing the concatenated string of relevant text from relevant nodes.
5. `QueryEvent` - Event containing both the relevant text and search text.
"""
logger.info("## Designing the Workflow")


class PrepEvent(Event):
    """Prep event (prepares for retrieval)."""

    pass


class RetrieveEvent(Event):
    """Retrieve event (gets retrieved nodes)."""

    retrieved_nodes: list[NodeWithScore]


class RelevanceEvalEvent(Event):
    """Relevance evaluation event (gets results of relevance evaluation)."""

    relevant_results: list[str]


class TextExtractEvent(Event):
    """Text extract event. Extracts relevant text and concatenates."""

    relevant_text: str


class QueryEvent(Event):
    """Query event. Queries given relevant text and search text."""

    relevant_text: str
    search_text: str


"""
Below is the code for the corrective RAG workflow:
"""
logger.info("Below is the code for the corrective RAG workflow:")


DEFAULT_RELEVANCY_PROMPT_TEMPLATE = PromptTemplate(
    template="""As a grader, your task is to evaluate the relevance of a document retrieved in response to a user's question.

    Retrieved Document:
    -------------------
    {context_str}

    User Question:
    --------------
    {query_str}

    Evaluation Criteria:
    - Consider whether the document contains keywords or topics related to the user's question.
    - The evaluation should not be overly stringent; the primary objective is to identify and filter out clearly irrelevant retrievals.

    Decision:
    - Assign a binary score to indicate the document's relevance.
    - Use 'yes' if the document is relevant to the question, or 'no' if it is not.

    Please provide your binary score ('yes' or 'no') below to indicate the document's relevance to the user question."""
)

DEFAULT_TRANSFORM_QUERY_TEMPLATE = PromptTemplate(
    template="""Your task is to refine a query to ensure it is highly effective for retrieving relevant search results. \n
    Analyze the given input to grasp the core semantic intent or meaning. \n
    Original Query:
    \n ------- \n
    {query_str}
    \n ------- \n
    Your goal is to rephrase or enhance this query to improve its search performance. Ensure the revised query is concise and directly aligned with the intended search objective. \n
    Respond with the optimized query only:"""
)


class CorrectiveRAGWorkflow(Workflow):
    @step
    async def ingest(self, ctx: Context, ev: StartEvent) -> StopEvent | None:
        """Ingest step (for ingesting docs and initializing index)."""
        documents: list[Document] | None = ev.get("documents")

        if documents is None:
            return None

        index = VectorStoreIndex.from_documents(documents)

        return StopEvent(result=index)

    @step
    async def prepare_for_retrieval(
        self, ctx: Context, ev: StartEvent
    ) -> PrepEvent | None:
        """Prepare for retrieval."""

        query_str: str | None = ev.get("query_str")
        retriever_kwargs: dict | None = ev.get("retriever_kwargs", {})

        if query_str is None:
            return None

        tavily_ai_apikey: str | None = ev.get("tavily_ai_apikey")
        index = ev.get("index")

        llm = MLXLlamaIndexLLMAdapter(
            model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats")

        async def run_async_code_58c0d29b():
            await ctx.store.set("llm", llm)
            return
         = asyncio.run(run_async_code_58c0d29b())
        logger.success(format_json())
        async def run_async_code_8375a763():
            await ctx.store.set("index", index)
            return 
         = asyncio.run(run_async_code_8375a763())
        logger.success(format_json())
        await ctx.store.set(
            "tavily_tool", TavilyToolSpec(api_key=tavily_ai_apikey)
        )

        async def run_async_code_de216c89():
            await ctx.store.set("query_str", query_str)
            return 
         = asyncio.run(run_async_code_de216c89())
        logger.success(format_json())
        async def run_async_code_0b52dd8b():
            await ctx.store.set("retriever_kwargs", retriever_kwargs)
            return 
         = asyncio.run(run_async_code_0b52dd8b())
        logger.success(format_json())

        return PrepEvent()

    @step
    async def retrieve(
        self, ctx: Context, ev: PrepEvent
    ) -> RetrieveEvent | None:
        """Retrieve the relevant nodes for the query."""
        async def run_async_code_e1ee7125():
            async def run_async_code_1b5d86fa():
                query_str = await ctx.store.get("query_str")
                return query_str
            query_str = asyncio.run(run_async_code_1b5d86fa())
            logger.success(format_json(query_str))
            return query_str
        query_str = asyncio.run(run_async_code_e1ee7125())
        logger.success(format_json(query_str))
        async def run_async_code_245d3900():
            async def run_async_code_b70d8cb9():
                retriever_kwargs = await ctx.store.get("retriever_kwargs")
                return retriever_kwargs
            retriever_kwargs = asyncio.run(run_async_code_b70d8cb9())
            logger.success(format_json(retriever_kwargs))
            return retriever_kwargs
        retriever_kwargs = asyncio.run(run_async_code_245d3900())
        logger.success(format_json(retriever_kwargs))

        if query_str is None:
            return None

        async def run_async_code_d89c866c():
            async def run_async_code_8c60d45f():
                index = await ctx.store.get("index", default=None)
                return index
            index = asyncio.run(run_async_code_8c60d45f())
            logger.success(format_json(index))
            return index
        index = asyncio.run(run_async_code_d89c866c())
        logger.success(format_json(index))
        async def run_async_code_04e4c597():
            async def run_async_code_4df4b6ca():
                tavily_tool = await ctx.store.get("tavily_tool", default=None)
                return tavily_tool
            tavily_tool = asyncio.run(run_async_code_4df4b6ca())
            logger.success(format_json(tavily_tool))
            return tavily_tool
        tavily_tool = asyncio.run(run_async_code_04e4c597())
        logger.success(format_json(tavily_tool))
        if not (index or tavily_tool):
            raise ValueError(
                "Index and tavily tool must be constructed. Run with 'documents' and 'tavily_ai_apikey' params first."
            )

        retriever: BaseRetriever = index.as_retriever(**retriever_kwargs)
        result = retriever.retrieve(query_str)
        async def run_async_code_a37abf48():
            await ctx.store.set("retrieved_nodes", result)
            return 
         = asyncio.run(run_async_code_a37abf48())
        logger.success(format_json())
        async def run_async_code_de216c89():
            await ctx.store.set("query_str", query_str)
            return 
         = asyncio.run(run_async_code_de216c89())
        logger.success(format_json())
        return RetrieveEvent(retrieved_nodes=result)

    @step
    async def eval_relevance(
        self, ctx: Context, ev: RetrieveEvent
    ) -> RelevanceEvalEvent:
        """Evaluate relevancy of retrieved documents with the query."""
        retrieved_nodes = ev.retrieved_nodes
        async def run_async_code_e1ee7125():
            async def run_async_code_1b5d86fa():
                query_str = await ctx.store.get("query_str")
                return query_str
            query_str = asyncio.run(run_async_code_1b5d86fa())
            logger.success(format_json(query_str))
            return query_str
        query_str = asyncio.run(run_async_code_e1ee7125())
        logger.success(format_json(query_str))

        relevancy_results = []
        for node in retrieved_nodes:
            async def run_async_code_9c783b73():
                async def run_async_code_726e8eec():
                    llm = await ctx.store.get("llm")
                    return llm
                llm = asyncio.run(run_async_code_726e8eec())
                logger.success(format_json(llm))
                return llm
            llm = asyncio.run(run_async_code_9c783b73())
            logger.success(format_json(llm))
            async def async_func_127():
                resp = llm.complete(
                    DEFAULT_RELEVANCY_PROMPT_TEMPLATE.format(
                        context_str=node.text, query_str=query_str
                    )
                )
                return resp
            resp = asyncio.run(async_func_127())
            logger.success(format_json(resp))
            relevancy_results.append(resp.text.lower().strip())

        async def run_async_code_fff5fb88():
            await ctx.store.set("relevancy_results", relevancy_results)
            return 
         = asyncio.run(run_async_code_fff5fb88())
        logger.success(format_json())
        return RelevanceEvalEvent(relevant_results=relevancy_results)

    @step
    async def extract_relevant_texts(
        self, ctx: Context, ev: RelevanceEvalEvent
    ) -> TextExtractEvent:
        """Extract relevant texts from retrieved documents."""
        async def run_async_code_e20a59b1():
            async def run_async_code_492f7791():
                retrieved_nodes = await ctx.store.get("retrieved_nodes")
                return retrieved_nodes
            retrieved_nodes = asyncio.run(run_async_code_492f7791())
            logger.success(format_json(retrieved_nodes))
            return retrieved_nodes
        retrieved_nodes = asyncio.run(run_async_code_e20a59b1())
        logger.success(format_json(retrieved_nodes))
        relevancy_results = ev.relevant_results

        relevant_texts = [
            retrieved_nodes[i].text
            for i, result in enumerate(relevancy_results)
            if result == "yes"
        ]

        result = "\n".join(relevant_texts)
        return TextExtractEvent(relevant_text=result)

    @step
    async def transform_query(
        self, ctx: Context, ev: TextExtractEvent
    ) -> QueryEvent:
        """Search the transformed query with Tavily API."""
        relevant_text = ev.relevant_text
        async def run_async_code_ece2580a():
            async def run_async_code_92568f0b():
                relevancy_results = await ctx.store.get("relevancy_results")
                return relevancy_results
            relevancy_results = asyncio.run(run_async_code_92568f0b())
            logger.success(format_json(relevancy_results))
            return relevancy_results
        relevancy_results = asyncio.run(run_async_code_ece2580a())
        logger.success(format_json(relevancy_results))
        async def run_async_code_e1ee7125():
            async def run_async_code_1b5d86fa():
                query_str = await ctx.store.get("query_str")
                return query_str
            query_str = asyncio.run(run_async_code_1b5d86fa())
            logger.success(format_json(query_str))
            return query_str
        query_str = asyncio.run(run_async_code_e1ee7125())
        logger.success(format_json(query_str))

        if "no" in relevancy_results:
            async def run_async_code_9c783b73():
                async def run_async_code_726e8eec():
                    llm = await ctx.store.get("llm")
                    return llm
                llm = asyncio.run(run_async_code_726e8eec())
                logger.success(format_json(llm))
                return llm
            llm = asyncio.run(run_async_code_9c783b73())
            logger.success(format_json(llm))
            async def async_func_165():
                resp = llm.complete(
                    DEFAULT_TRANSFORM_QUERY_TEMPLATE.format(query_str=query_str)
                )
                return resp
            resp = asyncio.run(async_func_165())
            logger.success(format_json(resp))
            transformed_query_str = resp.text
            async def run_async_code_4f0872c7():
                async def run_async_code_6451832f():
                    tavily_tool = await ctx.store.get("tavily_tool")
                    return tavily_tool
                tavily_tool = asyncio.run(run_async_code_6451832f())
                logger.success(format_json(tavily_tool))
                return tavily_tool
            tavily_tool = asyncio.run(run_async_code_4f0872c7())
            logger.success(format_json(tavily_tool))
            search_results = tavily_tool.search(
                transformed_query_str, max_results=5
            )
            search_text = "\n".join([result.text for result in search_results])
        else:
            search_text = ""

        return QueryEvent(relevant_text=relevant_text, search_text=search_text)

    @step
    async def query_result(self, ctx: Context, ev: QueryEvent) -> StopEvent:
        """Get result with relevant text."""
        relevant_text = ev.relevant_text
        search_text = ev.search_text
        async def run_async_code_e1ee7125():
            async def run_async_code_1b5d86fa():
                query_str = await ctx.store.get("query_str")
                return query_str
            query_str = asyncio.run(run_async_code_1b5d86fa())
            logger.success(format_json(query_str))
            return query_str
        query_str = asyncio.run(run_async_code_e1ee7125())
        logger.success(format_json(query_str))

        documents = [Document(text=relevant_text + "\n" + search_text)]
        index = SummaryIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        result = query_engine.query(query_str)
        return StopEvent(result=result)

"""
## Running the workflow
"""
logger.info("## Running the workflow")


documents = SimpleDirectoryReader("./data").load_data()
workflow = CorrectiveRAGWorkflow()
async def run_async_code_eb51c4f3():
    async def run_async_code_05224757():
        index = await workflow.run(documents=documents)
        return index
    index = asyncio.run(run_async_code_05224757())
    logger.success(format_json(index))
    return index
index = asyncio.run(run_async_code_eb51c4f3())
logger.success(format_json(index))


async def async_func_8():
    response = await workflow.run(
        query_str="How was Llama2 pretrained?",
        index=index,
        tavily_ai_apikey=tavily_ai_api_key,
    )
    return response
response = asyncio.run(async_func_8())
logger.success(format_json(response))
display(Markdown(str(response)))

async def async_func_15():
    response = await workflow.run(
        query_str="What is the functionality of latest ChatGPT memory."
    )
    return response
response = asyncio.run(async_func_15())
logger.success(format_json(response))
display(Markdown(str(response)))

logger.info("\n\n[DONE]", bright=True)