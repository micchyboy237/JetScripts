%pip install llama-index-llms-openai pandas[jinja2] spacyimport logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from llama_index.core import (
    TreeIndex,
    VectorStoreIndex,
    SimpleDirectoryReader,
    Response,
)
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import RelevancyEvaluator
from llama_index.core.node_parser import SentenceSplitter
import pandas as pd

pd.set_option("display.max_colwidth", 0)
gpt3 = OpenAI(temperature=0, model="gpt-3.5-turbo")

gpt4 = OpenAI(temperature=0, model="gpt-4")
evaluator = RelevancyEvaluator(llm=gpt3)
evaluator_gpt4 = RelevancyEvaluator(llm=gpt4)
documents = SimpleDirectoryReader("./test_wiki_data").load_data()
splitter = SentenceSplitter(chunk_size=512)
vector_index = VectorStoreIndex.from_documents(
    documents, transformations=[splitter]
)
from llama_index.core.evaluation import EvaluationResult


def display_eval_df(
    query: str, response: Response, eval_result: EvaluationResult
) -> None:
    eval_df = pd.DataFrame(
        {
            "Query": query,
            "Response": str(response),
            "Source": response.source_nodes[0].node.text[:1000] + "...",
            "Evaluation Result": "Pass" if eval_result.passing else "Fail",
            "Reasoning": eval_result.feedback,
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
query_str = (
    "What battles took place in New York City in the American Revolution?"
)
query_engine = vector_index.as_query_engine()
response_vector = query_engine.query(query_str)
eval_result = evaluator_gpt4.evaluate_response(
    query=query_str, response=response_vector
)
display_eval_df(query_str, response_vector, eval_result)
query_str = "What are the airports in New York City?"
query_engine = vector_index.as_query_engine()
response_vector = query_engine.query(query_str)
eval_result = evaluator_gpt4.evaluate_response(
    query=query_str, response=response_vector
)
display_eval_df(query_str, response_vector, eval_result)
query_str = "Who is the mayor of New York City?"
query_engine = vector_index.as_query_engine()
response_vector = query_engine.query(query_str)
eval_result = evaluator_gpt4.evaluate_response(
    query=query_str, response=response_vector
)
display_eval_df(query_str, response_vector, eval_result)
from typing import List


def display_eval_sources(
    query: str, response: Response, eval_result: List[str]
) -> None:
    sources = [s.node.get_text() for s in response.source_nodes]
    eval_df = pd.DataFrame(
        {
            "Source": sources,
            "Eval Result": eval_result,
        },
    )
    eval_df.style.set_caption(query)
    eval_df = eval_df.style.set_properties(
        **{
            "inline-size": "600px",
            "overflow-wrap": "break-word",
        },
        subset=["Source"]
    )

    display(eval_df)
query_str = "What are the airports in New York City?"
query_engine = vector_index.as_query_engine(
    similarity_top_k=3, response_mode="no_text"
)
response_vector = query_engine.query(query_str)
eval_source_result_full = [
    evaluator_gpt4.evaluate(
        query=query_str,
        response=response_vector.response,
        contexts=[source_node.get_content()],
    )
    for source_node in response_vector.source_nodes
]
eval_source_result = [
    "Pass" if result.passing else "Fail" for result in eval_source_result_full
]
display_eval_sources(query_str, response_vector, eval_source_result)
query_str = "Who is the mayor of New York City?"
query_engine = vector_index.as_query_engine(
    similarity_top_k=3, response_mode="no_text"
)
eval_source_result_full = [
    evaluator_gpt4.evaluate(
        query=query_str,
        response=response_vector.response,
        contexts=[source_node.get_content()],
    )
    for source_node in response_vector.source_nodes
]
eval_source_result = [
    "Pass" if result.passing else "Fail" for result in eval_source_result_full
]
display_eval_sources(query_str, response_vector, eval_source_result)
