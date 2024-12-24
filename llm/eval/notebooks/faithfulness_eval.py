# %pip install llama-index-llms-openai pandas[jinja2] spacy
import nest_asyncio

nest_asyncio.apply()
import os

os.environ["OPENAI_API_KEY"] = "sk-..."
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Response,
)
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.node_parser import SentenceSplitter
import pandas as pd

pd.set_option("display.max_colwidth", 0)
gpt4 = OpenAI(temperature=0, model="gpt-4")

evaluator_gpt4 = FaithfulnessEvaluator(llm=gpt4)
documents = SimpleDirectoryReader("./test_wiki_data/").load_data()
splitter = SentenceSplitter(chunk_size=512)
vector_index = VectorStoreIndex.from_documents(
    documents, transformations=[splitter]
)
from llama_index.core.evaluation import EvaluationResult


def display_eval_df(response: Response, eval_result: EvaluationResult) -> None:
    if response.source_nodes == []:
        print("no response!")
        return
    eval_df = pd.DataFrame(
        {
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
query_engine = vector_index.as_query_engine()
response_vector = query_engine.query("How did New York City get its name?")
eval_result = evaluator_gpt4.evaluate_response(response=response_vector)
display_eval_df(response_vector, eval_result)
from llama_index.core.evaluation import DatasetGenerator

question_generator = DatasetGenerator.from_documents(documents)
eval_questions = question_generator.generate_questions_from_nodes(5)

eval_questions
import asyncio


def evaluate_query_engine(query_engine, questions):
    c = [query_engine.aquery(q) for q in questions]
    results = asyncio.run(asyncio.gather(*c))
    print("finished query")

    total_correct = 0
    for r in results:
        eval_result = (
            1 if evaluator_gpt4.evaluate_response(response=r).passing else 0
        )
        total_correct += eval_result

    return total_correct, len(results)
vector_query_engine = vector_index.as_query_engine()
correct, total = evaluate_query_engine(vector_query_engine, eval_questions[:5])

print(f"score: {correct}/{total}")
