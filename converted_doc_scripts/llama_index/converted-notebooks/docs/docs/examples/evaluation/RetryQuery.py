from jet.logger import CustomLogger
from llama_index.core import Response
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.evaluation import GuidelineEvaluator
from llama_index.core.evaluation import RelevancyEvaluator
from llama_index.core.evaluation.guideline import DEFAULT_GUIDELINES
from llama_index.core.indices.query.query_transform.feedback_transform import (
FeedbackQueryTransformation,
)
from llama_index.core.query_engine import RetryGuidelineQueryEngine
from llama_index.core.query_engine import RetryQueryEngine
from llama_index.core.query_engine import RetrySourceQueryEngine
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/evaluation/RetryQuery.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Self Correcting Query Engines - Evaluation & Retry

In this notebook, we showcase several advanced, self-correcting query engines.  
They leverage the latest LLM's ability to evaluate its own output, and then self-correct to give better responses.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Self Correcting Query Engines - Evaluation & Retry")

# !pip install llama-index




"""
## Setup

First we ingest the document.
"""
logger.info("## Setup")


# import nest_asyncio

# nest_asyncio.apply()

"""
Download Data
"""
logger.info("Download Data")

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

"""
Load Data
"""
logger.info("Load Data")

documents = SimpleDirectoryReader("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/").load_data()
index = VectorStoreIndex.from_documents(documents)
query = "What did the author do growing up?"

"""
Let's what the response from the default query engine looks like
"""
logger.info("Let's what the response from the default query engine looks like")

base_query_engine = index.as_query_engine()
response = base_query_engine.query(query)
logger.debug(response)

"""
## Retry Query Engine

The retry query engine uses an evaluator to improve the response from a base query engine.  

It does the following:
1. first queries the base query engine, then
2. use the evaluator to decided if the response passes.
3. If the response passes, then return response,
4. Otherwise, transform the original query with the evaluation result (query, response, and feedback) into a new query, 
5. Repeat up to max_retries
"""
logger.info("## Retry Query Engine")


query_response_evaluator = RelevancyEvaluator()
retry_query_engine = RetryQueryEngine(
    base_query_engine, query_response_evaluator
)
retry_response = retry_query_engine.query(query)
logger.debug(retry_response)

"""
## Retry Source Query Engine

The Source Retry modifies the query source nodes by filtering the existing source nodes for the query based on llm node evaluation.
"""
logger.info("## Retry Source Query Engine")


retry_source_query_engine = RetrySourceQueryEngine(
    base_query_engine, query_response_evaluator
)
retry_source_response = retry_source_query_engine.query(query)
logger.debug(retry_source_response)

"""
## Retry Guideline Query Engine

This module tries to use guidelines to direct the evaluator's behavior. You can customize your own guidelines.
"""
logger.info("## Retry Guideline Query Engine")


guideline_eval = GuidelineEvaluator(
    guidelines=DEFAULT_GUIDELINES
    + "\nThe response should not be overly long.\n"
    "The response should try to summarize where possible.\n"
)  # just for example

"""
Let's look like what happens under the hood.
"""
logger.info("Let's look like what happens under the hood.")

typed_response = (
    response if isinstance(response, Response) else response.get_response()
)
eval = guideline_eval.evaluate_response(query, typed_response)
logger.debug(f"Guideline eval evaluation result: {eval.feedback}")

feedback_query_transform = FeedbackQueryTransformation(resynthesize_query=True)
transformed_query = feedback_query_transform.run(query, {"evaluation": eval})
logger.debug(f"Transformed query: {transformed_query.query_str}")

"""
Now let's run the full query engine
"""
logger.info("Now let's run the full query engine")

retry_guideline_query_engine = RetryGuidelineQueryEngine(
    base_query_engine, guideline_eval, resynthesize_query=True
)
retry_guideline_response = retry_guideline_query_engine.query(query)
logger.debug(retry_guideline_response)

logger.info("\n\n[DONE]", bright=True)