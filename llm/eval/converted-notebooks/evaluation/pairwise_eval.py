import re
from jet.llm.ollama import initialize_ollama_settings, create_llm
import pandas as pd
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.evaluation import PairwiseComparisonEvaluator
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Response
import sys
import logging
import nest_asyncio
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

# Pairwise Evaluator
#
# This notebook uses the `PairwiseEvaluator` module to see if an evaluation LLM would prefer one query engine over another.


nest_asyncio.apply()


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


initialize_ollama_settings()

pd.set_option("display.max_colwidth", 0)

# Using GPT-4 here for evaluation

gpt4 = create_llm(temperature=0, model="llama3.1")

evaluator_gpt4 = PairwiseComparisonEvaluator(llm=gpt4)

documents = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/summaries", required_exts=[".md"]).load_data()

splitter_512 = SentenceSplitter(chunk_size=512)
vector_index1 = VectorStoreIndex.from_documents(
    documents, transformations=[splitter_512]
)

splitter_200 = SentenceSplitter(chunk_size=200)
vector_index2 = VectorStoreIndex.from_documents(
    documents, transformations=[splitter_200]
)

query_engine1 = vector_index1.as_query_engine(similarity_top_k=2)
query_engine2 = vector_index2.as_query_engine(similarity_top_k=8)


def display_eval_df(query, response1, response2, eval_result) -> None:
    eval_results_dict = {
        **eval_result.model_dump(),
        "Reference Response (Answer 1)": response2 or "",
        "Current Response (Answer 2)": response1 or "",
    }
    eval_results_dict["feedback"] = eval_results_dict.pop("feedback")

    # Regular expression to match content between [[ and ]]
    pattern = r"\[\[(.*?)\]\]"

    # Find all matches
    matches = re.findall(pattern, eval_results_dict["feedback"])
    verdict = matches[-1] if matches else "No match"

    if verdict.lower() == 'c':
        verdict = "Tie"
    eval_results_dict["verdict"] = verdict

    logger.newline()
    logger.info("Eval Results:")
    items = [(key, result)
             for key, result in eval_results_dict.items()
             if result != None and key not in ["passing", "invalid_result", "response", "score"]]
    for key, result in items:
        if key == 'verdict':
            level = "SUCCESS"
            if result == "Tie":
                level = "WARNING"
            elif result == "No match":
                level = "ERROR"
            logger.log(f"{key.title()}:", result, colors=["DEBUG", level])
        else:
            logger.log(f"{key.title()}:", result, colors=["DEBUG", "SUCCESS"])


# To run evaluations you can call the `.evaluate_response()` function on the `Response` object return from the query to run the evaluations. Lets evaluate the outputs of the vector_index.

query_str = "Tell me about yourself."
response1 = str(query_engine1.query(query_str))
response2 = str(query_engine2.query(query_str))

# By default, we enforce "consistency" in the pairwise comparison.
#
# We try feeding in the candidate, reference pair, and then swap the order of the two, and make sure that the results are still consistent (or return a TIE if not).

eval_result = evaluator_gpt4.evaluate(
    query_str, response=response1, second_response=response2
)

logger.newline()
logger.info("Eval Query 1...")
display_eval_df(query_str, response1, response2, eval_result)

# **NOTE**: By default, we enforce consensus by flipping the order of response/reference and making sure that the answers are opposites.
#
# We can disable this - which can lead to more inconsistencies!

evaluator_gpt4_nc = PairwiseComparisonEvaluator(
    llm=gpt4, enforce_consensus=False
)

eval_result = evaluator_gpt4_nc.evaluate(
    query_str, response=response1, second_response=response2
)

logger.newline()
logger.info("Eval NC No Flip...")
display_eval_df(query_str, response1, response2, eval_result)

eval_result = evaluator_gpt4_nc.evaluate(
    query_str, response=response2, second_response=response1
)

logger.newline()
logger.info("Eval NC Flipped...")
display_eval_df(query_str, response2, response1, eval_result)

# Running on some more Queries

query_str = "What are your primary skills?"
response1 = str(query_engine1.query(query_str))
response2 = str(query_engine2.query(query_str))

eval_result = evaluator_gpt4.evaluate(
    query_str, response=response1, second_response=response2
)

logger.newline()
logger.info("Eval Query 2...")
display_eval_df(query_str, response1, response2, eval_result)

logger.info("\n\n[DONE]", bright=True)
