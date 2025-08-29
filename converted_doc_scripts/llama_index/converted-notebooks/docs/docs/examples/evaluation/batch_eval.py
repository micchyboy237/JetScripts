from jet.transformers.formatters import format_json
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Response
from llama_index.core.evaluation import (
FaithfulnessEvaluator,
RelevancyEvaluator,
CorrectnessEvaluator,
)
from llama_index.core.evaluation import BatchEvalRunner
from llama_index.core.evaluation import DatasetGenerator
from llama_index.core.node_parser import SentenceSplitter
import openai
import os
import pandas as pd
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# BatchEvalRunner - Running Multiple Evaluations

The `BatchEvalRunner` class can be used to run a series of evaluations asynchronously. The async jobs are limited to a defined size of `num_workers`.

## Setup
"""
logger.info("# BatchEvalRunner - Running Multiple Evaluations")

# %pip install llama-index-llms-ollama llama-index-embeddings-huggingface

# import nest_asyncio

# nest_asyncio.apply()


# os.environ["OPENAI_API_KEY"] = "sk-..."


pd.set_option("display.max_colwidth", 0)

"""
Using GPT-4 here for evaluation
"""
logger.info("Using GPT-4 here for evaluation")

gpt4 = OllamaFunctionCallingAdapter(temperature=0, model="llama3.2", request_timeout=300.0, context_window=4096)

faithfulness_gpt4 = FaithfulnessEvaluator(llm=gpt4)
relevancy_gpt4 = RelevancyEvaluator(llm=gpt4)
correctness_gpt4 = CorrectnessEvaluator(llm=gpt4)

documents = SimpleDirectoryReader("./test_wiki_data/").load_data()

llm = OllamaFunctionCallingAdapter(temperature=0.3, model="llama3.2", request_timeout=300.0, context_window=4096)
splitter = SentenceSplitter(chunk_size=512)
vector_index = VectorStoreIndex.from_documents(
    documents, transformations=[splitter]
)

"""
## Question Generation

To run evaluations in batch, you can create the runner and then call the `.aevaluate_queries()` function on a list of queries.

First, we can generate some questions and then run evaluation on them.
"""
logger.info("## Question Generation")

# %pip install spacy datasets span-marker scikit-learn


dataset_generator = DatasetGenerator.from_documents(documents, llm=llm)

qas = dataset_generator.generate_dataset_from_nodes(num=3)

"""
## Running Batch Evaluation

Now, we can run our batch evaluation!
"""
logger.info("## Running Batch Evaluation")


runner = BatchEvalRunner(
    {"faithfulness": faithfulness_gpt4, "relevancy": relevancy_gpt4},
    workers=8,
)

eval_results = runner.evaluate_queries(
        vector_index.as_query_engine(llm=llm), queries=qas.questions
    )
logger.success(format_json(eval_results))

logger.debug(len([qr for qr in qas.qr_pairs]))

"""
## Inspecting Outputs
"""
logger.info("## Inspecting Outputs")

logger.debug(eval_results.keys())

logger.debug(eval_results["faithfulness"][0].dict().keys())

logger.debug(eval_results["faithfulness"][0].passing)
logger.debug(eval_results["faithfulness"][0].response)
logger.debug(eval_results["faithfulness"][0].contexts)

"""
## Reporting Total Scores
"""
logger.info("## Reporting Total Scores")

def get_eval_results(key, eval_results):
    results = eval_results[key]
    correct = 0
    for result in results:
        if result.passing:
            correct += 1
    score = correct / len(results)
    logger.debug(f"{key} Score: {score}")
    return score

score = get_eval_results("faithfulness", eval_results)

score = get_eval_results("relevancy", eval_results)

logger.info("\n\n[DONE]", bright=True)