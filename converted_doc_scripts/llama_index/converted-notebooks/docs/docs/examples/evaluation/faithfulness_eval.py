from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from jet.logger import CustomLogger
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Response,
)
from llama_index.core.evaluation import DatasetGenerator
from llama_index.core.evaluation import EvaluationResult
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.node_parser import SentenceSplitter
import asyncio
import os
import pandas as pd
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Faithfulness Evaluator

This notebook uses the `FaithfulnessEvaluator` module to measure if the response from a query engine matches any source nodes.  
This is useful for measuring if the response was hallucinated.  
The data is extracted from the [New York City](https://en.wikipedia.org/wiki/New_York_City) wikipedia page.
"""
logger.info("# Faithfulness Evaluator")

# %pip install llama-index-llms-ollama pandas[jinja2] spacy

# import nest_asyncio

# nest_asyncio.apply()


# os.environ["OPENAI_API_KEY"] = "sk-..."


pd.set_option("display.max_colwidth", 0)

"""
Using GPT-4 here for evaluation
"""
logger.info("Using GPT-4 here for evaluation")

gpt4 = OllamaFunctionCallingAdapter(temperature=0, model="llama3.2")

evaluator_gpt4 = FaithfulnessEvaluator(llm=gpt4)

documents = SimpleDirectoryReader("./test_wiki_data/").load_data()

splitter = SentenceSplitter(chunk_size=512)
vector_index = VectorStoreIndex.from_documents(
    documents, transformations=[splitter]
)


def display_eval_df(response: Response, eval_result: EvaluationResult) -> None:
    if response.source_nodes == []:
        logger.debug("no response!")
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


"""
To run evaluations you can call the `.evaluate_response()` function on the `Response` object return from the query to run the evaluations. Lets evaluate the outputs of the vector_index.
"""
logger.info("To run evaluations you can call the `.evaluate_response()` function on the `Response` object return from the query to run the evaluations. Lets evaluate the outputs of the vector_index.")

query_engine = vector_index.as_query_engine()
response_vector = query_engine.query("How did New York City get its name?")
eval_result = evaluator_gpt4.evaluate_response(response=response_vector)

display_eval_df(response_vector, eval_result)

"""
## Benchmark on Generated Question

Now lets generate a few more questions so that we have more to evaluate with and run a small benchmark.
"""
logger.info("## Benchmark on Generated Question")


question_generator = DatasetGenerator.from_documents(documents)
eval_questions = question_generator.generate_questions_from_nodes(5)

eval_questions


def evaluate_query_engine(query_engine, questions):
    c = [query_engine.aquery(q) for q in questions]
    results = asyncio.run(asyncio.gather(*c))
    logger.debug("finished query")

    total_correct = 0
    for r in results:
        eval_result = (
            1 if evaluator_gpt4.evaluate_response(response=r).passing else 0
        )
        total_correct += eval_result

    return total_correct, len(results)


vector_query_engine = vector_index.as_query_engine()
correct, total = evaluate_query_engine(vector_query_engine, eval_questions[:5])

logger.debug(f"score: {correct}/{total}")

logger.info("\n\n[DONE]", bright=True)
