from jet.logger import CustomLogger
from llama_index.core import QueryBundle
from llama_index.core.tools import ToolMetadata
from llama_index.question_gen.openai import MLXQuestionGenerator
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/output_parsing/openai_sub_question.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# MLX function calling for Sub-Question Query Engine

In this notebook, we showcase how to use MLX function calling to improve the robustness of our sub-question query engine.

The sub-question query engine is designed to accept swappable question generators that implement the `BaseQuestionGenerator` interface.  
To leverage the power of openai function calling API, we implemented a new `MLXQuestionGenerator` (powered by our `MLXPydanticProgram`)

## MLX Question Generator

Unlike the default `LLMQuestionGenerator` that supports generic LLMs via the completion API, `MLXQuestionGenerator` only works with the latest MLX models that supports the function calling API. 

The benefit is that these models are fine-tuned to output JSON objects, so we can worry less about output parsing issues.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# MLX function calling for Sub-Question Query Engine")

# %pip install llama-index-question-gen-openai

# !pip install llama-index


question_gen = MLXQuestionGenerator.from_defaults()

"""
Let's test it out!
"""
logger.info("Let's test it out!")


tools = [
    ToolMetadata(
        name="march_22",
        description=(
            "Provides information about Uber quarterly financials ending March"
            " 2022"
        ),
    ),
    ToolMetadata(
        name="june_22",
        description=(
            "Provides information about Uber quarterly financials ending June"
            " 2022"
        ),
    ),
    ToolMetadata(
        name="sept_22",
        description=(
            "Provides information about Uber quarterly financials ending"
            " September 2022"
        ),
    ),
    ToolMetadata(
        name="sept_21",
        description=(
            "Provides information about Uber quarterly financials ending"
            " September 2022"
        ),
    ),
    ToolMetadata(
        name="june_21",
        description=(
            "Provides information about Uber quarterly financials ending June"
            " 2022"
        ),
    ),
    ToolMetadata(
        name="march_21",
        description=(
            "Provides information about Uber quarterly financials ending March"
            " 2022"
        ),
    ),
]

sub_questions = question_gen.generate(
    tools=tools,
    query=QueryBundle(
        "Compare the fastest growing sectors for Uber in the first two"
        " quarters of 2022"
    ),
)

sub_questions

logger.info("\n\n[DONE]", bright=True)