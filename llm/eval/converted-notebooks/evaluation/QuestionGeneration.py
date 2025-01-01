import json
import random
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Response
from llama_index.core.evaluation import DatasetGenerator, RelevancyEvaluator
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.evaluation import EvaluationResult
import pandas as pd
import sys
import logging
from script_utils import display_source_nodes
from jet.transformers import make_serializable
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
question_gen_query = f"You are a Job Employer. Your task is to setup {
    num_questions_per_chunk} questions for an upcoming interview. The questions should be relevant to the document. Restrict the questions to the context information provided."
question_generation_prompt = """\
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge.
generate only questions based on the below query.
Query: {query_str}
"""
question_generation_template = PromptTemplate(question_generation_prompt)

documents = reader.load_data()

data_generator = DatasetGenerator.from_documents(
    documents,
    num_questions_per_chunk=num_questions_per_chunk,
    question_gen_query=question_gen_query,
    text_question_template=question_generation_template,
)

eval_questions = data_generator.generate_questions_from_nodes()
eval_questions = random.sample(eval_questions, 5)

logger.newline()
logger.info("Generated eval questions:")
logger.success(json.dumps(make_serializable(eval_questions), indent=2))

gpt4 = Ollama(temperature=0, model="llama3.1",
              request_timeout=300.0, context_window=4096)

evaluator_gpt4 = RelevancyEvaluator(llm=gpt4)

vector_index = VectorStoreIndex.from_documents(documents)


def display_eval_df(query: str, response: Response, eval_result: EvaluationResult) -> None:
    display_source_nodes(query, response.source_nodes)

    logger.newline()
    logger.info("Eval Results:")
    items = [(key, result)
             for key, result in eval_result.model_dump().items() if result != None]
    for key, result in items:
        if key == 'passing':
            logger.log(f"{key.title()}:", "Passed" if result else "Failed", colors=[
                       "DEBUG", "SUCCESS" if result else "ERROR"])
        elif key == 'invalid_result':
            logger.log(f"{key.title()}:", "Valid" if not result else "Invalid", colors=[
                       "DEBUG", "SUCCESS" if not result else "ERROR"])
        else:
            logger.log(f"{key.title()}:", result, colors=["DEBUG", "SUCCESS"])


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
