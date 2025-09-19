async def main():
    from jet.transformers.formatters import format_json
    from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
    from jet.logger import CustomLogger
    from llama_index.core.workflow import (
        Workflow,
        step,
        StartEvent,
        StopEvent,
        Event,
        Context,
    )
    from llama_index.core.workflow.checkpointer import WorkflowCheckpointer
    import os
    import shutil

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    LOG_DIR = f"{OUTPUT_DIR}/logs"

    log_file = os.path.join(LOG_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.orange(f"Logs: {log_file}")

    """
    # Checkpointing Workflow Runs
    
    In this notebook, we demonstrate how to checkpoint `Workflow` runs via a `WorkflowCheckpointer` object. We also show how we can view all of the checkpoints that are stored in this object and finally how we can use a checkpoint as the starting point of a new run.
    
    ## Define a Workflow
    """
    logger.info("# Checkpointing Workflow Runs")

    # api_key = os.environ.get("OPENAI_API_KEY")

    class JokeEvent(Event):
        joke: str

    class JokeFlow(Workflow):
        llm = OllamaFunctionCalling(model="llama3.2")

        @step
        async def generate_joke(self, ev: StartEvent) -> JokeEvent:
            topic = ev.topic

            prompt = f"Write your best joke about {topic}."
            response = self.llm.complete(prompt)
            logger.success(format_json(response))
            return JokeEvent(joke=str(response))

        @step
        async def critique_joke(self, ev: JokeEvent) -> StopEvent:
            joke = ev.joke

            prompt = f"Give a thorough analysis and critique of the following joke: {joke}"
            response = self.llm.complete(prompt)
            logger.success(format_json(response))
            return StopEvent(result=str(response))

    """
    ## Define a WorkflowCheckpointer Object
    """
    logger.info("## Define a WorkflowCheckpointer Object")

    workflow = JokeFlow()
    wflow_ckptr = WorkflowCheckpointer(workflow=workflow)

    """
    ## Run the Workflow from the WorkflowCheckpointer
    
    The `WorkflowCheckpointer.run()` method is a wrapper over the `Workflow.run()` method, which injects a checkpointer callback in order to create and store checkpoints. Note that checkpoints are created at the completion of a step, and that the data stored in checkpoints are:
    
    - `last_completed_step`: The name of the last completed step
    - `input_event`: The input event to this last completed step
    - `output_event`: The event outputted by this last completed step
    - `ctx_state`: a snapshot of the attached `Context`
    """
    logger.info("## Run the Workflow from the WorkflowCheckpointer")

    handler = wflow_ckptr.run(
        topic="chemistry",
    )
    await handler

    """
    We can view all of the checkpoints via the `.checkpoints` attribute, which is dictionary with keys representing the `run_id` of the run and whose values are the list of checkpoints stored for the run.
    """
    logger.info("We can view all of the checkpoints via the `.checkpoints` attribute, which is dictionary with keys representing the `run_id` of the run and whose values are the list of checkpoints stored for the run.")

    wflow_ckptr.checkpoints

    for run_id, ckpts in wflow_ckptr.checkpoints.items():
        logger.debug(f"Run: {run_id} has {len(ckpts)} stored checkpoints")

    """
    ## Filtering the Checkpoints
    
    The `WorkflowCheckpointer` object also has a `.filter_checkpoints()` method that allows us to filter via:
    
    - The name of the last completed step by speciying the param `last_completed_step`
    - The event type of the last completed step's output event by specifying `output_event_type`
    - Similarly, the event type of the last completed step's input event by specifying `input_event_type`
    
    Specifying multiple of these filters will be combined by the "AND" operator.
    
    Let's test this functionality out, but first we'll make things a bit more interesting by running a couple of more runs with our `Workflow`.
    """
    logger.info("## Filtering the Checkpoints")

    additional_topics = ["biology", "history"]

    for topic in additional_topics:
        handler = wflow_ckptr.run(topic=topic)
        await handler

    for run_id, ckpts in wflow_ckptr.checkpoints.items():
        logger.debug(f"Run: {run_id} has {len(ckpts)} stored checkpoints")

    checkpoints_right_after_generate_joke_step = wflow_ckptr.filter_checkpoints(
        last_completed_step="generate_joke",
    )

    [ckpt for ckpt in checkpoints_right_after_generate_joke_step]

    """
    ## Re-Run Workflow from a specific checkpoint
    
    To run from a chosen `Checkpoint` we can use the `WorkflowCheckpointer.run_from()` method. NOTE that doing so will lead to a new `run` and it's checkpoints if enabled will be stored under the newly assigned `run_id`.
    """
    logger.info("## Re-Run Workflow from a specific checkpoint")

    new_workflow_instance = JokeFlow()
    wflow_ckptr.workflow = new_workflow_instance

    ckpt = checkpoints_right_after_generate_joke_step[0]

    handler = wflow_ckptr.run_from(checkpoint=ckpt)
    await handler

    for run_id, ckpts in wflow_ckptr.checkpoints.items():
        logger.debug(f"Run: {run_id} has {len(ckpts)} stored checkpoints")

    """
    Since we've executed from the checkpoint that represents the end of "generate_joke" step, there is only one additional checkpoint (i.e., that for the completion of step "critique_joke") that gets stored in the last partial run.
    
    ## Specifying Which Steps To Checkpoint
    
    By default all steps of the attached workflow (excluding the "_done" step) will be checkpointed. You can see which steps are enabled for checkpointing via the `enabled_checkpoints` attribute.
    """
    logger.info("## Specifying Which Steps To Checkpoint")

    wflow_ckptr.enabled_checkpoints

    """
    To disable a step for checkpointing, we can use the `.disable_checkpoint()` method
    """
    logger.info(
        "To disable a step for checkpointing, we can use the `.disable_checkpoint()` method")

    wflow_ckptr.disable_checkpoint(step="critique_joke")

    handler = wflow_ckptr.run(topic="cars")
    await handler

    for run_id, ckpts in wflow_ckptr.checkpoints.items():
        logger.debug(
            f"Run: {run_id} has stored checkpoints for steps {[c.last_completed_step for c in ckpts]}"
        )

    """
    And we can turn checkpointing back on by using the `.enable_checkpoint()` method
    """
    logger.info(
        "And we can turn checkpointing back on by using the `.enable_checkpoint()` method")

    wflow_ckptr.enable_checkpoint(step="critique_joke")

    handler = wflow_ckptr.run(topic="cars")
    await handler

    for run_id, ckpts in wflow_ckptr.checkpoints.items():
        logger.debug(
            f"Run: {run_id} has stored checkpoints for steps {[c.last_completed_step for c in ckpts]}"
        )

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
