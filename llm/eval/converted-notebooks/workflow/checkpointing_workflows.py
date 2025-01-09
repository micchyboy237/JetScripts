from llama_index.core.workflow.checkpointer import WorkflowCheckpointer
from jet.llm.ollama.base import Ollama
from llama_index.core.workflow import (
    Workflow,
    step,
    StartEvent,
    StopEvent,
    Event,
    Context,
)
import os
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# Checkpointing Workflow Runs

# In this notebook, we demonstrate how to checkpoint `Workflow` runs via a `WorkflowCheckpointer` object. We also show how we can view all of the checkpoints that are stored in this object and finally how we can use a checkpoint as the starting point of a new run.

# Define a Workflow


# api_key = os.environ.get("OPENAI_API_KEY")


class JokeEvent(Event):
    joke: str


class JokeFlow(Workflow):
    llm = Ollama(api_key=api_key)

    @step
    async def generate_joke(self, ev: StartEvent) -> JokeEvent:
        topic = ev.topic

        prompt = f"Write your best joke about {topic}."
        response = self.llm.complete(prompt)
        return JokeEvent(joke=str(response))

    @step
    async def critique_joke(self, ev: JokeEvent) -> StopEvent:
        joke = ev.joke

        prompt = f"Give a thorough analysis and critique of the following joke: {
            joke}"
        response = self.llm.complete(prompt)
        return StopEvent(result=str(response))

# Define a WorkflowCheckpointer Object


workflow = JokeFlow()
wflow_ckptr = WorkflowCheckpointer(workflow=workflow)

# Run the Workflow from the WorkflowCheckpointer

# The `WorkflowCheckpointer.run()` method is a wrapper over the `Workflow.run()` method, which injects a checkpointer callback in order to create and store checkpoints. Note that checkpoints are created at the completion of a step, and that the data stored in checkpoints are:
#
# - `last_completed_step`: The name of the last completed step
# - `input_event`: The input event to this last completed step
# - `output_event`: The event outputted by this last completed step
# - `ctx_state`: a snapshot of the attached `Context`

handler = wflow_ckptr.run(
    topic="chemistry",
)
await handler

# We can view all of the checkpoints via the `.checkpoints` attribute, which is dictionary with keys representing the `run_id` of the run and whose values are the list of checkpoints stored for the run.

wflow_ckptr.checkpoints

for run_id, ckpts in wflow_ckptr.checkpoints.items():
    print(f"Run: {run_id} has {len(ckpts)} stored checkpoints")

# Filtering the Checkpoints

# The `WorkflowCheckpointer` object also has a `.filter_checkpoints()` method that allows us to filter via:
#
# - The name of the last completed step by speciying the param `last_completed_step`
# - The event type of the last completed step's output event by specifying `output_event_type`
# - Similarly, the event type of the last completed step's input event by specifying `input_event_type`
#
# Specifying multiple of these filters will be combined by the "AND" operator.

# Let's test this functionality out, but first we'll make things a bit more interesting by running a couple of more runs with our `Workflow`.

additional_topics = ["biology", "history"]

for topic in additional_topics:
    handler = wflow_ckptr.run(topic=topic)
    await handler

for run_id, ckpts in wflow_ckptr.checkpoints.items():
    print(f"Run: {run_id} has {len(ckpts)} stored checkpoints")

checkpoints_right_after_generate_joke_step = wflow_ckptr.filter_checkpoints(
    last_completed_step="generate_joke",
)

[ckpt for ckpt in checkpoints_right_after_generate_joke_step]

# Re-Run Workflow from a specific checkpoint

# To run from a chosen `Checkpoint` we can use the `WorkflowCheckpointer.run_from()` method. NOTE that doing so will lead to a new `run` and it's checkpoints if enabled will be stored under the newly assigned `run_id`.

new_workflow_instance = JokeFlow()
wflow_ckptr.workflow = new_workflow_instance

ckpt = checkpoints_right_after_generate_joke_step[0]

handler = wflow_ckptr.run_from(checkpoint=ckpt)
await handler

for run_id, ckpts in wflow_ckptr.checkpoints.items():
    print(f"Run: {run_id} has {len(ckpts)} stored checkpoints")

# Since we've executed from the checkpoint that represents the end of "generate_joke" step, there is only one additional checkpoint (i.e., that for the completion of step "critique_joke") that gets stored in the last partial run.

# Specifying Which Steps To Checkpoint

# By default all steps of the attached workflow (excluding the "_done" step) will be checkpointed. You can see which steps are enabled for checkpointing via the `enabled_checkpoints` attribute.

wflow_ckptr.enabled_checkpoints

# To disable a step for checkpointing, we can use the `.disable_checkpoint()` method

wflow_ckptr.disable_checkpoint(step="critique_joke")

handler = wflow_ckptr.run(topic="cars")
await handler

for run_id, ckpts in wflow_ckptr.checkpoints.items():
    print(
        f"Run: {run_id} has stored checkpoints for steps {
            [c.last_completed_step for c in ckpts]}"
    )

# And we can turn checkpointing back on by using the `.enable_checkpoint()` method

wflow_ckptr.enable_checkpoint(step="critique_joke")

handler = wflow_ckptr.run(topic="cars")
await handler

for run_id, ckpts in wflow_ckptr.checkpoints.items():
    print(
        f"Run: {run_id} has stored checkpoints for steps {
            [c.last_completed_step for c in ckpts]}"
    )

logger.info("\n\n[DONE]", bright=True)
