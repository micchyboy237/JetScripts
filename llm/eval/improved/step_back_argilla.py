%pip install "argilla-llama-index>=2.1.0"from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.schema import NodeWithScore
from llama_index.core.workflow import (
    Context,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

from llama_index.core import get_response_synthesizer
from llama_index.core.workflow import Event
from llama_index.utils.workflow import draw_all_possible_flows
from llama_index.llms.openai import OpenAI

from argilla_llama_index import ArgillaHandler
import os

os.environ["OPENAI_API_KEY"] = "sk-..."
argilla_handler = ArgillaHandler(
    dataset_name="workflow_llama_index",
    api_url="http://localhost:6900",
    api_key="argilla.apikey",
    number_of_retrievals=2,
)
root_dispatcher = get_dispatcher()
root_dispatcher.add_span_handler(argilla_handler)
root_dispatcher.add_event_handler(argilla_handler)
class StepBackEvent(Event):
    """Get the step-back query"""

    step_back_query: str


class RetrieverEvent(Event):
    """Result of running the retrievals"""

    nodes_original: list[NodeWithScore]
    nodes_step_back: list[NodeWithScore]
STEP_BACK_TEMPLATE = """
You are an expert at world knowledge. Your task is to step back and
paraphrase a question to a more generic step-back question, which is
easier to answer. Here are a few examples:

Original Question: Which position did Knox Cunningham hold from May 1955 to Apr 1956?
Stepback Question: Which positions have Knox Cunningham held in his career?

Original Question: Who was the spouse of Anna Karina from 1968 to 1974?
Stepback Question: Who were the spouses of Anna Karina?

Original Question: what is the biggest hotel in las vegas nv as of November 28, 1993
Stepback Question: what is the size of the hotels in las vegas nv as of November 28, 1993?

Original Question: {original_query}
Stepback Question:
"""

GENERATE_ANSWER_TEMPLATE = """
You are an expert of world knowledge. I am going to ask you a question.
Your response should be comprehensive and not contradicted with the
following context if they are relevant. Otherwise, ignore them if they are
not relevant.

{context_original}
{context_step_back}

Original Question: {query}
Answer:
"""
class RAGWorkflow(Workflow):
    @step
    async def step_back(
        self, ctx: Context, ev: StartEvent
    ) -> StepBackEvent | None:
        """Generate the step-back query."""
        query = ev.get("query")
        index = ev.get("index")

        if not query:
            return None

        if not index:
            return None

        llm = Settings.llm
        step_back_query = llm.complete(
            prompt=STEP_BACK_TEMPLATE.format(original_query=query),
            formatted=True,
        )

        await ctx.set("query", query)
        await ctx.set("index", index)

        return StepBackEvent(step_back_query=str(step_back_query))

    @step
    async def retrieve(
        self, ctx: Context, ev: StepBackEvent
    ) -> RetrieverEvent | None:
        "Retrieve the relevant nodes for the original and step-back queries."
        query = await ctx.get("query", default=None)
        index = await ctx.get("index", default=None)

        await ctx.set("step_back_query", ev.step_back_query)

        retriever = index.as_retriever(similarity_top_k=2)
        nodes_step_back = await retriever.aretrieve(ev.step_back_query)
        nodes_original = await retriever.aretrieve(query)

        return RetrieverEvent(
            nodes_original=nodes_original, nodes_step_back=nodes_step_back
        )

    @step
    async def synthesize(self, ctx: Context, ev: RetrieverEvent) -> StopEvent:
        """Return a response using the contextualized prompt and retrieved nodes."""
        nodes_original = ev.nodes_original
        nodes_step_back = ev.nodes_step_back

        context_original = max(
            nodes_original, key=lambda node: node.get_score()
        ).get_text()
        context_step_back = max(
            nodes_step_back, key=lambda node: node.get_score()
        ).get_text()

        query = await ctx.get("query", default=None)
        formatted_query = GENERATE_ANSWER_TEMPLATE.format(
            context_original=context_original,
            context_step_back=context_step_back,
            query=query,
        )

        response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.COMPACT
        )

        response = response_synthesizer.synthesize(
            formatted_query, nodes=ev.nodes_original
        )
        return StopEvent(result=response)
draw_all_possible_flows(RAGWorkflow, filename="step_back_workflow.html")
!mkdir -p ../../data
!curl https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt -o ../../data/paul_graham_essay.txtSettings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.8)

transformations = [
    SentenceSplitter(chunk_size=256, chunk_overlap=75),
]

documents = SimpleDirectoryReader("../../data").load_data()
index = VectorStoreIndex.from_documents(
    documents=documents,
    transformations=transformations,
)
w = RAGWorkflow()

result = await w.run(query="What's Paul's work", index=index)
result
