from llama_index.core.evaluation import BatchEvalRunner
from llama_index.core.evaluation import DatasetGenerator
from jet.llm.ollama import initialize_ollama_settings, create_llm
import pandas as pd
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
)
from jet.llm.ollama.base import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Response
import os
import nest_asyncio
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# BatchEvalRunner - Running Multiple Evaluations
#
# The `BatchEvalRunner` class can be used to run a series of evaluations asynchronously. The async jobs are limited to a defined size of `num_workers`.
#
# Setup


nest_asyncio.apply()


initialize_ollama_settings()

pd.set_option("display.max_colwidth", 0)

# Using GPT-4 here for evaluation

gpt4 = Ollama(temperature=0, model="llama3.1")

faithfulness_gpt4 = FaithfulnessEvaluator(llm=gpt4)
relevancy_gpt4 = RelevancyEvaluator(llm=gpt4)
correctness_gpt4 = CorrectnessEvaluator(llm=gpt4)

documents = SimpleDirectoryReader("./test_wiki_data/").load_data()
documents = documents[:1]

splitter = SentenceSplitter(chunk_size=512)
vector_index = VectorStoreIndex.from_documents(
    documents, transformations=[splitter]
)

# Question Generation
#
# To run evaluations in batch, you can create the runner and then call the `.aevaluate_queries()` function on a list of queries.
#
# First, we can generate some questions and then run evaluation on them.


llm = create_llm(temperature=0.3, model="llama3.1")

dataset_generator = DatasetGenerator.from_documents(
    documents, llm=llm, show_progress=True, num_questions_per_chunk=3)

qas = dataset_generator.generate_dataset_from_nodes(num=3)

qas.questions

qas.qr_pairs

# Running Batch Evaluation
#
# Now, we can run our batch evaluation!


runner = BatchEvalRunner(
    {"faithfulness": faithfulness_gpt4, "relevancy": relevancy_gpt4},
    workers=4,
)

eval_results = runner.evaluate_queries(
    vector_index.as_query_engine(llm=llm), queries=qas.questions
)

# Inspecting Outputs

print(eval_results.keys())

print(eval_results["faithfulness"][0].dict().keys())

print(eval_results["faithfulness"][0].passing)
print(eval_results["faithfulness"][0].response)
print(eval_results["faithfulness"][0].contexts)

# Reporting Total Scores


def get_eval_results(key, eval_results):
    results = eval_results[key]
    correct = 0
    for result in results:
        if result.passing:
            correct += 1
    score = correct / len(results)
    print(f"{key} Score: {score}")
    return score


score = get_eval_results("faithfulness", eval_results)

score = get_eval_results("relevancy", eval_results)

logger.info("\n\n[DONE]", bright=True)
