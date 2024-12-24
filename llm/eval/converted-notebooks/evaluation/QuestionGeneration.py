from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Response
from llama_index.core.evaluation import DatasetGenerator, RelevancyEvaluator
import pandas as pd
import sys
import logging
from script_utils import display_source_nodes
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/evaluation/QuestionGeneration.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# QuestionGeneration
#
# This notebook walks through the process of generating a list of questions that could be asked about your data. This is useful for setting up an evaluation pipeline using the `FaithfulnessEvaluator` and `RelevancyEvaluator` evaluation tools.

# If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.

# %pip install llama-index-llms-ollama

# !pip install llama-index


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# Download Data

# !mkdir -p 'data/paul_graham/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

# Load Data

reader = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/JetScripts/llm/eval/converted-notebooks/retrievers/data/jet-resume", required_exts=[".md"])
num_questions_per_chunk = 3

documents = reader.load_data()

data_generator = DatasetGenerator.from_documents(
    documents, num_questions_per_chunk=num_questions_per_chunk)

eval_questions = data_generator.generate_questions_from_nodes()

eval_questions

gpt4 = Ollama(temperature=0, model="llama3.1",
              request_timeout=300.0, context_window=4096)

evaluator_gpt4 = RelevancyEvaluator(llm=gpt4)

vector_index = VectorStoreIndex.from_documents(documents)


def display_eval_df(query: str, response: Response, eval_result: str) -> None:
    response = str(response)
    passed = eval_result.passing
    feedback = eval_result.feedback

    display_source_nodes(query, response.source_nodes)
    if passed:
        logger.log("Result:", "Passed", colors=["GRAY", "SUCCESS"])
    else:
        logger.log("Result:", "Failed", colors=["GRAY", "ERROR"])
    logger.log("Feedback:", feedback, colors=["GRAY", "INFO"])

    eval_df = eval_df.style.set_properties(
        **{
            "inline-size": "600px",
            "overflow-wrap": "break-word",
        },
        subset=["Response", "Source"]
    )
    # Print the DataFrame as a table
    logger.debug("eval_df:")
    logger.success(eval_df.to_string())


for idx, question in enumerate(eval_questions):
    logger.newline()
    logger.info(f"# Question {idx + 1} eval:")
    query_engine = vector_index.as_query_engine()
    response_vector = query_engine.query(question)
    eval_result = evaluator_gpt4.evaluate_response(
        query=question, response=response_vector
    )

    display_eval_df(question, response_vector, eval_result)

logger.info("\n\n[DONE]", bright=True)
