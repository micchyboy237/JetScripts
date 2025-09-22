from jet.transformers.formatters import format_json
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import Ollama
from jet.logger import logger
from llama_index.core.workflow import (
StartEvent,
StopEvent,
Workflow,
step,
Event,
Context,
)
from llama_index.utils.workflow import draw_all_possible_flows
import asyncio
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
---
sidebar:
  order: 5
---

# Streaming events

Workflows can be complex -- they are designed to handle complex, branching, concurrent logic -- which means they can take time to fully execute. To provide your user with a good experience, you may want to provide an indication of progress by streaming events as they occur. Workflows have built-in support for this on the `Context` object.

To get this done, let's bring in all the deps we need:
"""
logger.info("# Streaming events")


"""
Let's set up some events for a simple three-step workflow, plus an event to handle streaming our progress as we go:
"""
logger.info("Let's set up some events for a simple three-step workflow, plus an event to handle streaming our progress as we go:")

class FirstEvent(Event):
    first_output: str


class SecondEvent(Event):
    second_output: str
    response: str


class ProgressEvent(Event):
    msg: str

"""
And define a workflow class that sends events:
"""
logger.info("And define a workflow class that sends events:")

class MyWorkflow(Workflow):
    @step
    async def step_one(self, ctx: Context, ev: StartEvent) -> FirstEvent:
        ctx.write_event_to_stream(ProgressEvent(msg="Step one is happening"))
        return FirstEvent(first_output="First step complete.")

    @step
    async def step_two(self, ctx: Context, ev: FirstEvent) -> SecondEvent:
        llm = Ollama(model="llama3.2")
        generator = llm.stream_complete(
                "Please give me the first 3 paragraphs of Moby Dick, a book in the public domain."
            )
        logger.success(format_json(generator))
        async for response in generator:
            ctx.write_event_to_stream(ProgressEvent(msg=response.delta))
        return SecondEvent(
            second_output="Second step complete, full response attached",
            response=str(response),
        )

    @step
    async def step_three(self, ctx: Context, ev: SecondEvent) -> StopEvent:
        ctx.write_event_to_stream(ProgressEvent(msg="Step three is happening"))
        return StopEvent(result="Workflow complete.")

"""
<Aside type="tip">
# `Ollama()` here assumes you have an `OPENAI_API_KEY` set in your environment. You could also pass one in using the `api_key` parameter.
</Aside>

In `step_one` and `step_three` we write individual events to the event stream. In `step_two` we use `astream_complete` to produce an iterable generator of the LLM's response, then we produce an event for each chunk of data the LLM sends back to us -- roughly one per word -- before returning the final response to `step_three`.

To actually get this output, we need to run the workflow asynchronously and listen for the events, like this:
"""
logger.info("In `step_one` and `step_three` we write individual events to the event stream. In `step_two` we use `astream_complete` to produce an iterable generator of the LLM's response, then we produce an event for each chunk of data the LLM sends back to us -- roughly one per word -- before returning the final response to `step_three`.")

async def main():
    w = MyWorkflow(timeout=30, verbose=True)
    handler = w.run(first_input="Start the workflow.")

    async for ev in handler.stream_events():
        if isinstance(ev, ProgressEvent):
            logger.debug(ev.msg)

    final_result = await handler
    logger.success(format_json(final_result))
    logger.debug("Final result", final_result)

    draw_all_possible_flows(MyWorkflow, filename="streaming_workflow.html")


if __name__ == "__main__":
    asyncio.run(main())

"""
`run` runs the workflow in the background, while `stream_events` will provide any event that gets written to the stream. It stops when the stream delivers a `StopEvent`, after which you can get the final result of the workflow as you normally would.


Next let's look at [concurrent execution](/python/framework/understanding/workflows/concurrent_execution).
"""
logger.info("Next let's look at [concurrent execution](/python/framework/understanding/workflows/concurrent_execution).")

logger.info("\n\n[DONE]", bright=True)