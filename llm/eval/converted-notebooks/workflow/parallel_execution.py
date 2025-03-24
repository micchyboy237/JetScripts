import time
import random
from llama_index.core.workflow.checkpointer import WorkflowCheckpointer
from llama_index.core.workflow import (
    step,
    Context,
    Workflow,
    Event,
    StartEvent,
    StopEvent,
)
import asyncio
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

# Parallel Execution of Same Event Example
#
# In this example, we'll demonstrate how to use the workflow functionality to achieve similar capabilities while allowing parallel execution of multiple events of the same type.
# By setting the `num_workers` parameter in `@step` decorator, we can control the number of steps executed simultaneously, enabling efficient parallel processing.
#

#
# Installing Dependencies
#
# First, we need to install the necessary dependencies:
#
# * LlamaIndex core for most functionalities
# * llama-index-utils-workflow for workflow capabilities


# Importing Required Libraries
# After installing the dependencies, we can import the required libraries:


# We will create two workflows: one that can process multiple data items in parallel by using the `@step(num_workers=N)` decorator, and another without setting num_workers, for comparison.
# By using the `num_workers` parameter in the `@step` decorator, we can limit the number of steps executed simultaneously, thus controlling the level of parallelism. This approach is particularly suitable for scenarios that require processing similar tasks while managing resource usage.
# For example, you can execute multiple sub-queries at once, but please note that num_workers cannot be set without limits. It depends on  your workload or token limits.
# Defining Event Types
# We'll define two event types: one for input events to be processed, and another for processing results:


class ProcessEvent(Event):
    data: str


class ResultEvent(Event):
    result: str

# Creating Sequential and Parallel Workflows
# Now, we'll create a SequentialWorkflow and a ParallelWorkflow class that includes three main steps:
#
# - start: Initialize and send multiple parallel events
# - process_data: Process data
# - combine_results: Collect and merge all processing results


class SequentialWorkflow(Workflow):
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> ProcessEvent:
        data_list = ["A", "B", "C"]
        await ctx.set("num_to_collect", len(data_list))
        for item in data_list:
            ctx.send_event(ProcessEvent(data=item))
        return None

    @step(num_workers=1)
    async def process_data(self, ev: ProcessEvent) -> ResultEvent:
        processing_time = 2 + random.random()
        await asyncio.sleep(processing_time)
        result = f"Processed: {ev.data}"
        print(f"Completed processing: {ev.data}")
        return ResultEvent(result=result)

    @step
    async def combine_results(
        self, ctx: Context, ev: ResultEvent
    ) -> StopEvent | None:
        num_to_collect = await ctx.get("num_to_collect")
        results = ctx.collect_events(ev, [ResultEvent] * num_to_collect)
        if results is None:
            return None

        combined_result = ", ".join([event.result for event in results])
        return StopEvent(result=combined_result)


class ParallelWorkflow(Workflow):
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> ProcessEvent:
        data_list = ["A", "B", "C"]
        await ctx.set("num_to_collect", len(data_list))
        for item in data_list:
            ctx.send_event(ProcessEvent(data=item))
        return None

    @step(num_workers=3)
    async def process_data(self, ev: ProcessEvent) -> ResultEvent:
        processing_time = 2 + random.random()
        await asyncio.sleep(processing_time)
        result = f"Processed: {ev.data}"
        print(f"Completed processing: {ev.data}")
        return ResultEvent(result=result)

    @step
    async def combine_results(
        self, ctx: Context, ev: ResultEvent
    ) -> StopEvent | None:
        num_to_collect = await ctx.get("num_to_collect")
        results = ctx.collect_events(ev, [ResultEvent] * num_to_collect)
        if results is None:
            return None

        combined_result = ", ".join([event.result for event in results])
        return StopEvent(result=combined_result)

# In these two workflows:
#
# - The start method initializes and sends multiple ProcessEvent.
# - The process_data method uses
#   - only the `@step` decorator in SequentialWorkflow
#   - uses the `@step(num_workers=3)` decorator in ParallelWorkflow to limit the number of simultaneously executing workers to 3.
# - The combine_results method collects all processing results and merges them.
#
# Running the Workflow
# Finally, we can create a main function to run our workflow:


sequential_workflow = SequentialWorkflow()

print(
    "Start a sequential workflow without setting num_workers in the step of process_data"
)
start_time = time.time()
result = await sequential_workflow.run()
end_time = time.time()
print(f"Workflow result: {result}")
print(f"Time taken: {end_time - start_time} seconds")
print("-" * 30)

parallel_workflow = ParallelWorkflow()

print(
    "Start a parallel workflow with setting num_workers in the step of process_data"
)
start_time = time.time()
result = await parallel_workflow.run()
end_time = time.time()
print(f"Workflow result: {result}")
print(f"Time taken: {end_time - start_time} seconds")

# Note
#
# - Without setting `num_workers=1`, it might take a total of 6-9 seconds. By setting `num_workers=3`, the processing occurs in parallel, handling 3 items at a time, and only takes 2-3 seconds total.
# - In ParallelWorkflow, the order of the completed results may differ from the input order, depending on the completion time of the tasks.

# This example demonstrates the execution speed with and without using num_workers, and how to implement parallel processing in a workflow. By setting num_workers, we can control the degree of parallelism, which is very useful for scenarios that need to balance performance and resource usage.

# Checkpointing

# Checkpointing a parallel execution Workflow like the one defined above is also possible. To do so, we must wrap the `Workflow` with a `WorkflowCheckpointer` object and perfrom the runs with these instances. During the execution of the workflow, checkpoints are stored in this wrapper object and can be used for inspection and as starting points for run executions.


wflow_ckptr = WorkflowCheckpointer(workflow=parallel_workflow)
handler = wflow_ckptr.run()
await handler

# Checkpoints for the above run are stored in the `WorkflowCheckpointer.checkpoints` Dict attribute.

for run_id, ckpts in wflow_ckptr.checkpoints.items():
    print(f"Run: {run_id} has {[c.last_completed_step for c in ckpts]}")

# We can run from any of the checkpoints stored, using `WorkflowCheckpointer.run_from(checkpoint=...)` method. Let's take the first checkpoint that was stored after the first completion of "process_data" and run from it.

ckpt = wflow_ckptr.checkpoints[run_id][0]
handler = wflow_ckptr.run_from(ckpt)
await handler

# Invoking a `run_from` or `run` will create a new run entry in the `checkpoints` attribute. In the latest run from the specified checkpoint, we can see that only two more "process_data" steps and the final "combine_results" steps were left to be completed.

for run_id, ckpts in wflow_ckptr.checkpoints.items():
    print(f"Run: {run_id} has {[c.last_completed_step for c in ckpts]}")

# Now, if we use the checkpoint associated with the second completion of "process_data" of the same initial run as the starting point, then we should see a new entry that only has two steps: "process_data" and "combine_results".

first_run_id = next(iter(wflow_ckptr.checkpoints.keys()))
first_run_id

ckpt = wflow_ckptr.checkpoints[first_run_id][
    1
]  # checkpoint after the second "process_data" step
handler = wflow_ckptr.run_from(ckpt)
await handler

for run_id, ckpts in wflow_ckptr.checkpoints.items():
    print(f"Run: {run_id} has {[c.last_completed_step for c in ckpts]}")

# Similarly, if we start with the checkpoint for the third completion of "process_data" of the initial run, then we should only see the final "combine_results" step.

ckpt = wflow_ckptr.checkpoints[first_run_id][
    2
]  # checkpoint after the third "process_data" step
handler = wflow_ckptr.run_from(ckpt)
await handler

for run_id, ckpts in wflow_ckptr.checkpoints.items():
    print(f"Run: {run_id} has {[c.last_completed_step for c in ckpts]}")

logger.info("\n\n[DONE]", bright=True)
