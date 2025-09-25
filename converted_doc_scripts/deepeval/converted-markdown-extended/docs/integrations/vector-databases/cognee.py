from jet.transformers.formatters import format_json
from cognee.api.v1.search import SearchType
from deepeval import evaluate
from deepeval.metrics import (
from deepeval.test_case import LLMTestCase
from jet.logger import logger
import cognee
import os
import shutil

async def main():
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric,
    )
    
    
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
    id: cognee
    title: Cognee
    sidebar_label: Cognee
    ---
    
    ## Quick Summary
    
    Cognee is an open-source framework for anyone to easily implement graph RAG into their LLM application. You can learn more by visiting their [website here.](https://www.cognee.ai/)
    
    :::info
    With Cognee, you should see an increase in your [`ContextualRelevancyMetric`](/docs/metrics-contextual-relevancy), [`ContextualRecallMetric`](/docs/metrics-contextual-recall), and [`ContextualPrecisionMetric`](/docs/metrics-contextual-precision) scores.
    :::
    
    Unlike traditional vector databases that relies on simple embedding retrieval and re-rankings to retrieve `retrieval_context`s, Cognee stores and creates a "semantic graph" out of your data, which allows for more accurate retrievals.
    
    ## Setup Cognee
    
    Simply add your LLM API key to the environment variables:
    """
    logger.info("## Quick Summary")
    
    # os.environ["LLM_API_KEY"] = "YOUR_OPENAI_API_KEY"
    
    """
    For those on Networkx, you can also create an account on Graphistry to visualize results:
    """
    logger.info("For those on Networkx, you can also create an account on Graphistry to visualize results:")
    
    
    cognee.config.set_graphistry_config({
        "username": "YOUR_USERNAME",
        "password": "YOUR_PASSWORD"
    })
    
    """
    Finally, ingest your data into Cognee and run some retrievals:
    """
    logger.info("Finally, ingest your data into Cognee and run some retrievals:")
    
    
    ...
    text = "Cognee is the Graph RAG Framework"
    await cognee.add(text) # add a new piece of information
    await cognee.cognify() # create a semantic graph using cognee
    
    retrieval_context = await cognee.search(SearchType.INSIGHTS, query_text="What is Cognee?")
    logger.success(format_json(retrieval_context))
    for context in retrieval_context:
        logger.debug(context)
    
    """
    ## Evaluating Cognee RAG Pipelines
    
    Unit testing RAG pipelines powered by Cognee is as simple as defining an `EvaluationDataset` and generating `actual_output`s and `retrieval_context`s at evaluation time. Building upon the previous example, first generate all the necessarily parameters required to test RAG:
    """
    logger.info("## Evaluating Cognee RAG Pipelines")
    
    ...
    
    input = "What is Cognee?"
    retrieval_context = await cognee.search(SearchType.INSIGHTS, query_text="What is Cognee?")
    logger.success(format_json(retrieval_context))
    
    prompt = """
    Answer the user question based on the supporting context
    
    User Question:
    {input}
    
    Supporting Context:
    {retrieval_context}
    """
    
    actual_output = generate(prompt) # hypothetical function, replace with your own LLM
    
    """
    Then, simply run `evaluate()`:
    """
    logger.info("Then, simply run `evaluate()`:")
    
    
    ...
    test_case = LLMTestCase(
        input=input,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
        expected_output="Cognee is the Graph RAG Framework.",
    )
    evaluate(
        [test_case],
        metrics=[
            ContextualRecallMetric(),
            ContextualPrecisionMetric(),
            ContextualRelevancyMetric(),
        ],
    )
    
    """
    That's it! Do you notice an increase in the contextual metric scores?
    """
    logger.info("That's it! Do you notice an increase in the contextual metric scores?")
    
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