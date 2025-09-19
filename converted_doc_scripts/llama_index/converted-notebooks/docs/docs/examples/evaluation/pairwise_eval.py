from jet.transformers.formatters import format_json
from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
from jet.logger import CustomLogger
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Response
from llama_index.core.evaluation import PairwiseComparisonEvaluator
from llama_index.core.node_parser import SentenceSplitter
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
# Pairwise Evaluator

This notebook uses the `PairwiseEvaluator` module to see if an evaluation LLM would prefer one query engine over another.
"""
logger.info("# Pairwise Evaluator")

# %pip install llama-index-llms-ollama

# import nest_asyncio

# nest_asyncio.apply()


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


pd.set_option("display.max_colwidth", 0)

"""
Using GPT-4 here for evaluation
"""
logger.info("Using GPT-4 here for evaluation")

gpt4 = OllamaFunctionCalling(temperature=0, model="llama3.2")

evaluator_gpt4 = PairwiseComparisonEvaluator(llm=gpt4)

documents = SimpleDirectoryReader("./test_wiki_data/").load_data()

splitter_512 = SentenceSplitter(chunk_size=512)
vector_index1 = VectorStoreIndex.from_documents(
    documents, transformations=[splitter_512]
)

splitter_128 = SentenceSplitter(chunk_size=128)
vector_index2 = VectorStoreIndex.from_documents(
    documents, transformations=[splitter_128]
)

query_engine1 = vector_index1.as_query_engine(similarity_top_k=2)
query_engine2 = vector_index2.as_query_engine(similarity_top_k=8)


def display_eval_df(query, response1, response2, eval_result) -> None:
    eval_df = pd.DataFrame(
        {
            "Query": query,
            "Reference Response (Answer 1)": response2,
            "Current Response (Answer 2)": response1,
            "Score": eval_result.score,
            "Reason": eval_result.feedback,
        },
        index=[0],
    )
    eval_df = eval_df.style.set_properties(
        **{
            "inline-size": "300px",
            "overflow-wrap": "break-word",
        },
        subset=["Current Response (Answer 2)", "Reference Response (Answer 1)"]
    )
    display(eval_df)


"""
To run evaluations you can call the `.evaluate_response()` function on the `Response` object return from the query to run the evaluations. Lets evaluate the outputs of the vector_index.
"""
logger.info("To run evaluations you can call the `.evaluate_response()` function on the `Response` object return from the query to run the evaluations. Lets evaluate the outputs of the vector_index.")

query_str = "What was the role of NYC during the American Revolution?"
response1 = str(query_engine1.query(query_str))
response2 = str(query_engine2.query(query_str))

"""
By default, we enforce "consistency" in the pairwise comparison.

We try feeding in the candidate, reference pair, and then swap the order of the two, and make sure that the results are still consistent (or return a TIE if not).
"""
logger.info("By default, we enforce "consistency" in the pairwise comparison.")

eval_result = evaluator_gpt4.evaluate(
    query_str, response=response1, reference=response2
)
logger.success(format_json(eval_result))

display_eval_df(query_str, response1, response2, eval_result)

"""
**NOTE**: By default, we enforce consensus by flipping the order of response/reference and making sure that the answers are opposites.

We can disable this - which can lead to more inconsistencies!
"""
logger.info("We can disable this - which can lead to more inconsistencies!")

evaluator_gpt4_nc = PairwiseComparisonEvaluator(
    llm=gpt4, enforce_consensus=False
)

eval_result = evaluator_gpt4_nc.evaluate(
    query_str, response=response1, reference=response2
)
logger.success(format_json(eval_result))

display_eval_df(query_str, response1, response2, eval_result)

eval_result = evaluator_gpt4_nc.evaluate(
    query_str, response=response2, reference=response1
)
logger.success(format_json(eval_result))

display_eval_df(query_str, response2, response1, eval_result)

"""
## Running on some more Queries
"""
logger.info("## Running on some more Queries")

query_str = "Tell me about the arts and culture of NYC"
response1 = str(query_engine1.query(query_str))
response2 = str(query_engine2.query(query_str))

eval_result = evaluator_gpt4.evaluate(
    query_str, response=response1, reference=response2
)
logger.success(format_json(eval_result))

display_eval_df(query_str, response1, response2, eval_result)

logger.info("\n\n[DONE]", bright=True)
