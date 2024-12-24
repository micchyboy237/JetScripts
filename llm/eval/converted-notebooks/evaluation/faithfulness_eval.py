import httpx
import asyncio
from llama_index.core.evaluation import DatasetGenerator
from llama_index.core.evaluation import EvaluationResult
import pandas as pd
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.llms.ollama import Ollama
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Response,
)
import nest_asyncio
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
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
    "/Users/jethroestrada/Desktop/External_Projects/JetScripts/llm/eval/converted-notebooks/retrievers/data/jet-resume", required_exts=[".md"]).load_data()
num_questions_per_chunk = 3

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
    # Print the DataFrame as a table
    logger.debug("eval_df:")
    logger.success(eval_df.to_string(index=False))

# To run evaluations you can call the `.evaluate_response()` function on the `Response` object return from the query to run the evaluations. Lets evaluate the outputs of the vector_index.


query = "Tell me about yourself."
query_engine = vector_index.as_query_engine()
response_vector = query_engine.query(query)
eval_result = evaluator_gpt4.evaluate_response(response=response_vector)

display_eval_df(response_vector, eval_result)

# Benchmark on Generated Question
#
# Now lets generate a few more questions so that we have more to evaluate with and run a small benchmark.


question_generator = DatasetGenerator.from_documents(
    documents, num_questions_per_chunk=num_questions_per_chunk)
eval_questions = question_generator.generate_questions_from_nodes(5)

eval_questions


TIMEOUT = httpx.Timeout(120.0, connect=10.0)  # Adjust as needed
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
