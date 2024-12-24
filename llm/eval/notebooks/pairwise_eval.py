# %pip install llama-index-llms-openai
import nest_asyncio

nest_asyncio.apply()
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Response
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import PairwiseComparisonEvaluator
from llama_index.core.node_parser import SentenceSplitter
import pandas as pd

pd.set_option("display.max_colwidth", 0)
gpt4 = OpenAI(temperature=0, model="gpt-4")

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
query_str = "What was the role of NYC during the American Revolution?"
response1 = str(query_engine1.query(query_str))
response2 = str(query_engine2.query(query_str))
eval_result = await evaluator_gpt4.aevaluate(
    query_str, response=response1, reference=response2
)
display_eval_df(query_str, response1, response2, eval_result)
evaluator_gpt4_nc = PairwiseComparisonEvaluator(
    llm=gpt4, enforce_consensus=False
)
eval_result = await evaluator_gpt4_nc.aevaluate(
    query_str, response=response1, reference=response2
)
display_eval_df(query_str, response1, response2, eval_result)
eval_result = await evaluator_gpt4_nc.aevaluate(
    query_str, response=response2, reference=response1
)
display_eval_df(query_str, response2, response1, eval_result)
query_str = "Tell me about the arts and culture of NYC"
response1 = str(query_engine1.query(query_str))
response2 = str(query_engine2.query(query_str))
eval_result = await evaluator_gpt4.aevaluate(
    query_str, response=response1, reference=response2
)
display_eval_df(query_str, response1, response2, eval_result)
