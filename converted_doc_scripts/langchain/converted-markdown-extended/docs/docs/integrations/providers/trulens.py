from jet.logger import logger
from trulens_eval import Tru
from trulens_eval import TruChain
from trulens_eval.feedback import Feedback, Huggingface,
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
# TruLens

>[TruLens](https://trulens.org) is an [open-source](https://github.com/truera/trulens) package that provides instrumentation and evaluation tools for large language model (LLM) based applications.

This page covers how to use [TruLens](https://trulens.org) to evaluate and track LLM apps built on langchain.


## Installation and Setup

Install the `trulens-eval` python package.
"""
logger.info("# TruLens")

pip install trulens-eval

"""
## Quickstart

See the integration details in the [TruLens documentation](https://www.trulens.org/trulens_eval/getting_started/quickstarts/langchain_quickstart/).

### Tracking

Once you've created your LLM chain, you can use TruLens for evaluation and tracking.
TruLens has a number of [out-of-the-box Feedback Functions](https://www.trulens.org/trulens_eval/evaluation/feedback_functions/),
and is also an extensible framework for LLM evaluation.

Create the feedback functions:
"""
logger.info("## Quickstart")


hugs = Huggingface()
ollama = Ollama()

lang_match = Feedback(hugs.language_match).on_input_output()

qa_relevance = Feedback(ollama.relevance).on_input_output()

toxicity = Feedback(ollama.toxicity).on_input()

"""
### Chains

After you've set up Feedback Function(s) for evaluating your LLM, you can wrap your application with
TruChain to get detailed tracing, logging and evaluation of your LLM app.

Note: See code for the `chain` creation is in
the [TruLens documentation](https://www.trulens.org/trulens_eval/getting_started/quickstarts/langchain_quickstart/).
"""
logger.info("### Chains")


truchain = TruChain(
    chain,
    app_id='Chain1_ChatApplication',
    feedbacks=[lang_match, qa_relevance, toxicity]
)
truchain("que hora es?")

"""
### Evaluation

Now you can explore your LLM-based application!

Doing so will help you understand how your LLM application is performing at a glance. As you iterate new versions of your LLM application, you can compare their performance across all of the different quality metrics you've set up. You'll also be able to view evaluations at a record level, and explore the chain metadata for each record.
"""
logger.info("### Evaluation")


tru = Tru()
tru.run_dashboard() # open a Streamlit app to explore

"""
For more information on TruLens, visit [trulens.org](https://www.trulens.org/)
"""
logger.info("For more information on TruLens, visit [trulens.org](https://www.trulens.org/)")

logger.info("\n\n[DONE]", bright=True)