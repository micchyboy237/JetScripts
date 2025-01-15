from jet.llm.ollama import Ollama
from llama_index.utils.workflow import draw_most_recent_execution
from llama_index.core.workflow import draw_all_possible_flows
import random
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context,
)
import os
import asyncio
from jet.transformers.formatters import format_json
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

file_name = os.path.splitext(os.path.basename(__file__))[0]
generated_dir = os.path.join("results", file_name)
os.makedirs(generated_dir, exist_ok=True)


class InputEvent(Event):
    input: str


class SetupEvent(Event):
    error: bool


class QueryEvent(Event):
    query: str


class PromptEnhancerWorkflow(Workflow):
    @step
    async def setup(self, ctx: Context, ev: StartEvent) -> SetupEvent:
        if not hasattr(self, "initialized") or not self.initialized:
            self.initialized = True
            logger.debug("I got initialized")
        return SetupEvent(error=False)

    @step
    async def collect_input(self, ev: StartEvent) -> InputEvent:
        if hasattr(ev, "input"):
            logger.debug("I got some input")
            return InputEvent(input=ev.input)

    @step
    async def parse_query(self, ev: StartEvent) -> QueryEvent:
        if hasattr(ev, "query"):
            logger.debug("I got a query")
            return QueryEvent(query=ev.query)

    @step
    async def run_query(
        self, ctx: Context, ev: InputEvent | SetupEvent | QueryEvent
    ) -> StopEvent | None:
        ready = ctx.collect_events(ev, [QueryEvent, InputEvent, SetupEvent])
        if ready is None:
            logger.debug("Not enough events yet")
            return None

        logger.debug("Now I have all the events")
        logger.debug(ready)

        result = f"Ran query '{ready[0].query}' on input '{ready[1].input}'"
        return StopEvent(result=result)


c = PromptEnhancerWorkflow()


async def run_async_code_b9804d94():
    result = await c.run(input="Here's some input", query="Here's my question")
    return result
result = asyncio.run(run_async_code_b9804d94())
logger.success(format_json(result))

draw_all_possible_flows(PromptEnhancerWorkflow, f"{
                        generated_dir}/prompt_enhancer_workflow.html")
