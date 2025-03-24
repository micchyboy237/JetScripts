# %pip install llama-index-llms-openai pandas[jinja2] spacy
import httpx
import asyncio
from llama_index.core.evaluation import DatasetGenerator
from llama_index.core.evaluation import EvaluationResult
from jet.llm.ollama.base import initialize_ollama_settings
import pandas as pd
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Response,
)
import os
import nest_asyncio

nest_asyncio.apply()

# os.environ["OPENAI_API_KEY"] = "sk-..."
# from llama_index.llms.openai import OpenAI

initialize_ollama_settings()

pd.set_option("display.max_colwidth", 0)
llm = Ollama(temperature=0, model="llama3.1")

evaluator_gpt4 = FaithfulnessEvaluator(llm=llm)
documents = SimpleDirectoryReader("./test_wiki_data/").load_data()
splitter = SentenceSplitter(chunk_size=512)
vector_index = VectorStoreIndex.from_documents(
    documents, transformations=[splitter]
)


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

question_generator = DatasetGenerator.from_documents(documents)
eval_questions = question_generator.generate_questions_from_nodes(5)

eval_questions


# Set a custom timeout
TIMEOUT = httpx.Timeout(300.0, connect=10.0)  # Adjust as needed
client = httpx.AsyncClient(timeout=TIMEOUT)


async def evaluate_query_engine(query_engine, questions):
    async with client:  # Ensure proper cleanup
        total_correct = 0
        total = 0

        for question in questions:
            try:
                # Process each query
                response = await query_engine.aquery(question)
                eval_response = evaluator_gpt4.evaluate_response(
                    response=response)
                eval_result = (
                    1 if eval_response.passing else 0
                )
                total_correct += eval_result
                total += 1

                # Yield progress
                yield {"question": question, "correct": total_correct, "total": total}

            except Exception as e:
                # Handle errors
                yield {"question": question, "error": str(e), "correct": total_correct, "total": total}


vector_query_engine = vector_index.as_query_engine()
eval_questions_sample = eval_questions[:5]  # Example questions

async for progress in evaluate_query_engine(vector_query_engine, eval_questions_sample):
    if "error" in progress:
        print(f"Question: {progress['question']} - Error: {progress['error']}")
    else:
        print(f"Progress: {progress['correct']}/{progress['total']}")
