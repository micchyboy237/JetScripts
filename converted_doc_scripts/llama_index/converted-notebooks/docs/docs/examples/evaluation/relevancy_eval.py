from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import (
TreeIndex,
VectorStoreIndex,
SimpleDirectoryReader,
Response,
)
from llama_index.core.evaluation import EvaluationResult
from llama_index.core.evaluation import RelevancyEvaluator
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import List
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

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# Relevancy Evaluator

This notebook uses the `RelevancyEvaluator` to measure if the response + source nodes match the query.  
This is useful for measuring if the query was actually answered by the response.
"""
logger.info("# Relevancy Evaluator")

# %pip install llama-index-llms-ollama pandas[jinja2] spacy


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


pd.set_option("display.max_colwidth", 0)

gpt3 = MLXLlamaIndexLLMAdapter(temperature=0, model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats")

gpt4 = MLXLlamaIndexLLMAdapter(temperature=0, model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats")

evaluator = RelevancyEvaluator(llm=gpt3)
evaluator_gpt4 = RelevancyEvaluator(llm=gpt4)

documents = SimpleDirectoryReader("./test_wiki_data").load_data()

splitter = SentenceSplitter(chunk_size=512)
vector_index = VectorStoreIndex.from_documents(
    documents, transformations=[splitter]
)



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

"""
### Evaluate Response

Evaluate response relative to source nodes as well as query.
"""
logger.info("### Evaluate Response")

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

"""
### Evaluate Source Nodes

Evaluate the set of returned sources, and determine which sources actually contain the answer to a given query.
"""
logger.info("### Evaluate Source Nodes")



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

logger.info("\n\n[DONE]", bright=True)