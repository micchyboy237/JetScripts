from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core.evaluation import GuidelineEvaluator
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/evaluation/guideline_eval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Guideline Evaluator

This notebook shows how to use `GuidelineEvaluator` to evaluate a question answer system given user specified guidelines.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Guideline Evaluator")

# %pip install llama-index-llms-ollama

# !pip install llama-index


# import nest_asyncio

# nest_asyncio.apply()

GUIDELINES = [
    "The response should fully answer the query.",
    "The response should avoid being vague or ambiguous.",
    (
        "The response should be specific and use statistics or numbers when"
        " possible."
    ),
]

llm = OllamaFunctionCallingAdapter(model="llama3.2")

evaluators = [
    GuidelineEvaluator(llm=llm, guidelines=guideline)
    for guideline in GUIDELINES
]

sample_data = {
    "query": "Tell me about global warming.",
    "contexts": [
        (
            "Global warming refers to the long-term increase in Earth's"
            " average surface temperature due to human activities such as the"
            " burning of fossil fuels and deforestation."
        ),
        (
            "It is a major environmental issue with consequences such as"
            " rising sea levels, extreme weather events, and disruptions to"
            " ecosystems."
        ),
        (
            "Efforts to combat global warming include reducing carbon"
            " emissions, transitioning to renewable energy sources, and"
            " promoting sustainable practices."
        ),
    ],
    "response": (
        "Global warming is a critical environmental issue caused by human"
        " activities that lead to a rise in Earth's temperature. It has"
        " various adverse effects on the planet."
    ),
}

for guideline, evaluator in zip(GUIDELINES, evaluators):
    eval_result = evaluator.evaluate(
        query=sample_data["query"],
        contexts=sample_data["contexts"],
        response=sample_data["response"],
    )
    logger.debug("=====")
    logger.debug(f"Guideline: {guideline}")
    logger.debug(f"Pass: {eval_result.passing}")
    logger.debug(f"Feedback: {eval_result.feedback}")

logger.info("\n\n[DONE]", bright=True)
