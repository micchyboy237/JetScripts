import json
import httpx
import asyncio
from llama_index.core.evaluation import DatasetGenerator
from llama_index.core.evaluation import EvaluationResult
import pandas as pd
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.evaluation import FaithfulnessEvaluator
from jet.llm.ollama.base import Ollama
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Response,
)
import nest_asyncio
import random
from llama_index.core.prompts.base import PromptTemplate
from jet.transformers.object import make_serializable
from jet.llm.utils.llama_index_utils import display_jet_source_nodes
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

# Faithfulness Evaluator
#
# This notebook uses the `FaithfulnessEvaluator` module to measure if the response from a query engine matches any source nodes.
# This is useful for measuring if the response was hallucinated.
# The data is extracted from the [New York City](https://en.wikipedia.org/wiki/New_York_City) wikipedia page.


nest_asyncio.apply()


initialize_ollama_settings()

pd.set_option("display.max_colwidth", 0)

# Using GPT-4 here for evaluation

llm = Ollama(temperature=0, model="llama3.1")

evaluator_gpt4 = FaithfulnessEvaluator(llm=llm)

documents = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data", required_exts=[".md"]).load_data()
num_questions_per_chunk = 3
question_gen_query = f"You are a Job Employer. Your task is to setup {
    num_questions_per_chunk} questions for an upcoming interview. The questions should be relevant to the document. Restrict the questions to the context information provided."
question_generation_prompt = """\
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge.
generate only questions based on the below query.
Query: {query_str}
"""
question_generation_template = PromptTemplate(question_generation_prompt)

splitter = SentenceSplitter(chunk_size=512)
vector_index = VectorStoreIndex.from_documents(
    documents, transformations=[splitter]
)


def display_eval_df(query: str, response: Response, eval_result: EvaluationResult) -> None:
    display_jet_source_nodes(query, response.source_nodes)

    logger.newline()
    logger.info("Eval Results:")
    items = [(key, result)
             for key, result in eval_result.model_dump().items() if result != None]
    for key, result in items:
        if key == 'passing':
            logger.log(f"{key.title()}:", "Passed" if result else "Failed", colors=[
                       "DEBUG", "SUCCESS" if result else "ERROR"])
        elif key == 'invalid_result':
            logger.log(f"{key.title()}:", "Valid" if not result else "Invalid", colors=[
                       "DEBUG", "SUCCESS" if not result else "ERROR"])
        else:
            logger.log(f"{key.title()}:", result, colors=["DEBUG", "SUCCESS"])

# To run evaluations you can call the `.evaluate_response()` function on the `Response` object return from the query to run the evaluations. Lets evaluate the outputs of the vector_index.


query = "Tell me about yourself."
query_engine = vector_index.as_query_engine()
response_vector = query_engine.query(query)
eval_result = evaluator_gpt4.evaluate_response(response=response_vector)

display_eval_df(query, response_vector, eval_result)

# Benchmark on Generated Question
#
# Now lets generate a few more questions so that we have more to evaluate with and run a small benchmark.


question_generator = DatasetGenerator.from_documents(
    documents,
    num_questions_per_chunk=num_questions_per_chunk,
    question_gen_query=question_gen_query,
    text_question_template=question_generation_template,
)
eval_questions = question_generator.generate_questions_from_nodes()

logger.newline()
logger.info(f"Generated eval questions ({len(eval_questions)}):")
logger.success(json.dumps(make_serializable(eval_questions), indent=2))


TIMEOUT = httpx.Timeout(300.0, connect=10.0)  # Adjust as needed
client = httpx.AsyncClient(timeout=TIMEOUT)


async def evaluate_query_engine(query_engine, questions):
    async with client:  # Ensure proper cleanup
        total_correct = 0
        total = 0

        for question in questions:
            try:
                response = query_engine.query(question)
                eval_response = evaluator_gpt4.evaluate_response(
                    response=response)
                eval_result = (
                    1 if eval_response.passing else 0
                )
                total_correct += eval_result
                total += 1

                yield {
                    "question": question,
                    "response": eval_response.response,
                    "correct": total_correct,
                    "total": total,
                    "passing": eval_response.passing,
                    "contexts": eval_response.contexts,
                    "feedback": eval_response.feedback,
                    "score": eval_response.score,
                    "query": eval_response.query,
                }

            except Exception as e:
                yield {"question": question, "error": str(e), "correct": total_correct, "total": total}


async def run_inline():
    vector_query_engine = vector_index.as_query_engine()
    eval_questions_sample = eval_questions[:5]  # Example questions

    async for progress in evaluate_query_engine(vector_query_engine, eval_questions_sample):
        if "error" in progress:
            print(f"Question: {progress['question']
                               } - Error: {progress['error']}")
        else:
            print(f"Progress: {progress['correct']}/{progress['total']}")

    logger.info("\n\n[DONE]", bright=True)

# Running the inline async function
asyncio.run(run_inline())

logger.info("\n\n[DONE]", bright=True)
