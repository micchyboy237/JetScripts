from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
from jet.logger import CustomLogger
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Response
from llama_index.core.evaluation import DatasetGenerator, RelevancyEvaluator
import logging
import os
import pandas as pd
import shutil
import sys


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/evaluation/QuestionGeneration.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# QuestionGeneration

This notebook walks through the process of generating a list of questions that could be asked about your data. This is useful for setting up an evaluation pipeline using the `FaithfulnessEvaluator` and `RelevancyEvaluator` evaluation tools.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# QuestionGeneration")

# %pip install llama-index-llms-ollama

# !pip install llama-index


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


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

reader = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data/")
documents = reader.load_data()

data_generator = DatasetGenerator.from_documents(documents)

eval_questions = data_generator.generate_questions_from_nodes()

eval_questions

gpt4 = OllamaFunctionCalling(temperature=0, model="llama3.2")

evaluator_gpt4 = RelevancyEvaluator(llm=gpt4)

vector_index = VectorStoreIndex.from_documents(documents)


def display_eval_df(query: str, response: Response, eval_result: str) -> None:
    eval_df = pd.DataFrame(
        {
            "Query": query,
            "Response": str(response),
            "Source": (
                response.source_nodes[0].node.get_content()[:1000] + "..."
            ),
            "Evaluation Result": eval_result,
        },
        index=[0],
    )
    eval_df = eval_df.style.set_properties(
        **{
            "inline-size": "600px",
            "overflow-wrap": "break-word",
        },
        subset=["Response", "Source"]
    )
    display(eval_df)


query_engine = vector_index.as_query_engine()
response_vector = query_engine.query(eval_questions[1])
eval_result = evaluator_gpt4.evaluate_response(
    query=eval_questions[1], response=response_vector
)

display_eval_df(eval_questions[1], response_vector, eval_result)

logger.info("\n\n[DONE]", bright=True)
