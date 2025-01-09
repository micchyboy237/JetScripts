import nest_asyncio
from jet.llm.ollama.base import Ollama
from llama_index.core.evaluation import GuidelineEvaluator
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/evaluation/guideline_eval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Guideline Evaluator

# This notebook shows how to use `GuidelineEvaluator` to evaluate a question answer system given user specified guidelines.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.


initialize_ollama_settings()


nest_asyncio.apply()

GUIDELINES = [
    "The response should fully answer the query.",
    "The response should avoid being vague or ambiguous.",
    (
        "The response should be specific and use statistics or numbers when"
        " possible."
    ),
]

llm = Ollama(temperature=0, model="llama3.1")

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
    print("=====")
    print(f"Guideline: {guideline}")
    print(f"Pass: {eval_result.passing}")
    print(f"Feedback: {eval_result.feedback}")

logger.info("\n\n[DONE]", bright=True)
