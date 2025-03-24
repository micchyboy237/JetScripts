from jet.llm.utils.llama_index_utils import display_jet_source_node
from llama_index.core.text_splitter import SentenceSplitter
from IPython.display import display, HTML
import os
from jet.vectors import SettingsManager, IndexManager
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Response,
)
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.evaluation import (
    DatasetGenerator,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
    RetrieverEvaluator,
    generate_question_context_pairs,
)
import pandas as pd
import sys
import logging
import nest_asyncio
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

# Evaluating RAG Systems
#
# Evaluation and benchmarking are crucial in developing LLM applications. Optimizing performance for applications like RAG (Retrieval Augmented Generation) requires a robust measurement mechanism.
#
# LlamaIndex provides essential modules to assess the quality of generated outputs and evaluate content retrieval quality. It categorizes its evaluation into two main types:
#
# *   **Response Evaluation** : Assesses quality of Generated Outputs
# *   **Retrieval Evaluation** : Assesses Retrieval quality
#
# [Documentation
# ](https://docs.llamaindex.ai/en/latest/module_guides/evaluating/)


# Response Evaluation
#
# Evaluating results from LLMs is distinct from traditional machine learning's straightforward outcomes. LlamaIndex employs evaluation modules, using a benchmark LLM like GPT-4, to gauge answer accuracy. Notably, these modules often blend query, context, and response, minimizing the need for ground-truth labels.
#
# The evaluation modules manifest in the following categories:
#
# *   **Faithfulness:** Assesses whether the response remains true to the retrieved contexts, ensuring there's no distortion or "hallucination."
# *   **Relevancy:** Evaluates the relevance of both the retrieved context and the generated answer to the initial query.
# *   **Correctness:** Determines if the generated answer aligns with the reference answer based on the query (this does require labels).
#
# Furthermore, LlamaIndex has the capability to autonomously generate questions from your data, paving the way for an evaluation pipeline to assess the RAG application.


nest_asyncio.apply()


logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set logger level to INFO

logger.handlers = []

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)  # Set handler level to INFO

logger.addHandler(handler)


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


settings_manager = SettingsManager.create({
    "llm_model": "llama3.1",
    "embedding_model": "nomic-embed-text",
    "chunk_size": 768,
    "chunk_overlap": 50,
})


# Download Data


# Load Data

reader = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/data/paul_graham")
documents = reader.load_data()

# Generate Question


base_nodes = IndexManager.create_nodes(
    documents=documents, parser=settings_manager.node_parser)

dataset_generator = DatasetGenerator(
    base_nodes[:20],
    llm=settings_manager.llm,
    show_progress=True,
    num_questions_per_chunk=3,
)

eval_dataset = dataset_generator.generate_dataset_from_nodes(num=5)
eval_dataset

eval_queries = list(eval_dataset.queries.values())

(eval_queries)

len(eval_queries)

# To be consistent we will fix evaluation query

eval_query = "How did the author describe their early attempts at writing short stories?"

gpt35 = SettingsManager.create_llm(
    model="llama3.2",
    base_url="http://localhost:11434",
    temperature=0,
)

gpt4 = SettingsManager.create_llm(
    model="llama3.1",
    base_url="http://localhost:11434",
    temperature=0,
)

vector_index = VectorStoreIndex.from_documents(documents, llm=gpt35)

query_engine = vector_index.as_query_engine()

retriever = vector_index.as_retriever(similarity_top_k=3)
nodes = retriever.retrieve(eval_query)


display(HTML(f'<p style="font-size:20px">{nodes[1].get_text()}</p>'))

# Faithfullness Evaluator
#
#  Measures if the response from a query engine matches any source nodes. This is useful for measuring if the response was hallucinated.

faithfulness_evaluator = FaithfulnessEvaluator(llm=gpt4)

response_vector = query_engine.query(eval_query)

eval_result = faithfulness_evaluator.evaluate_response(
    response=response_vector
)

eval_result.passing

eval_result

# Relevency Evaluation
#
# Measures if the response + source nodes match the query.

relevancy_evaluator = RelevancyEvaluator(llm=gpt4)

response_vector = query_engine.query(eval_query)

eval_result = relevancy_evaluator.evaluate_response(
    query=eval_query, response=response_vector
)

eval_result.query

eval_result.response

eval_result.passing

# Relevancy evaluation with multiple source nodes.

query_engine = vector_index.as_query_engine(similarity_top_k=3)

response_vector = query_engine.query(eval_query)

eval_source_result_full = [
    relevancy_evaluator.evaluate(
        query=eval_query,
        response=response_vector.response,
        contexts=[source_node.get_content()],
    )
    for source_node in response_vector.source_nodes
]

eval_source_result = [
    "Pass" if result.passing else "Fail" for result in eval_source_result_full
]

eval_source_result

# Correctness Evaluator
#
# Evaluates the relevance and correctness of a generated answer against a reference answer.

correctness_evaluator = CorrectnessEvaluator(llm=gpt4)

query = "Can you explain the theory of relativity proposed by Albert Einstein in detail?"

reference = """
Certainly! Albert Einstein's theory of relativity consists of two main components: special relativity and general relativity. Special relativity, published in 1905, introduced the concept that the laws of physics are the same for all non-accelerating observers and that the speed of light in a vacuum is a constant, regardless of the motion of the source or observer. It also gave rise to the famous equation E=mc², which relates energy (E) and mass (m).

General relativity, published in 1915, extended these ideas to include the effects of gravity. According to general relativity, gravity is not a force between masses, as described by Newton's theory of gravity, but rather the result of the warping of space and time by mass and energy. Massive objects, such as planets and stars, cause a curvature in spacetime, and smaller objects follow curved paths in response to this curvature. This concept is often illustrated using the analogy of a heavy ball placed on a rubber sheet, causing it to create a depression that other objects (representing smaller masses) naturally move towards.

In essence, general relativity provided a new understanding of gravity, explaining phenomena like the bending of light by gravity (gravitational lensing) and the precession of the orbit of Mercury. It has been confirmed through numerous experiments and observations and has become a fundamental theory in modern physics.
"""

response = """
Certainly! Albert Einstein's theory of relativity consists of two main components: special relativity and general relativity. Special relativity, published in 1905, introduced the concept that the laws of physics are the same for all non-accelerating observers and that the speed of light in a vacuum is a constant, regardless of the motion of the source or observer. It also gave rise to the famous equation E=mc², which relates energy (E) and mass (m).

However, general relativity, published in 1915, extended these ideas to include the effects of magnetism. According to general relativity, gravity is not a force between masses but rather the result of the warping of space and time by magnetic fields generated by massive objects. Massive objects, such as planets and stars, create magnetic fields that cause a curvature in spacetime, and smaller objects follow curved paths in response to this magnetic curvature. This concept is often illustrated using the analogy of a heavy ball placed on a rubber sheet with magnets underneath, causing it to create a depression that other objects (representing smaller masses) naturally move towards due to magnetic attraction.
"""

correctness_result = correctness_evaluator.evaluate(
    query=query,
    response=response,
    reference=reference,
)

correctness_result

correctness_result.score

correctness_result.passing

correctness_result.feedback

# Retrieval Evaluation
#
# Evaluates the quality of any Retriever module defined in LlamaIndex.
#
# To assess the quality of a Retriever module in LlamaIndex, we use metrics like hit-rate and MRR. These compare retrieved results to ground-truth context for any question. For simpler evaluation dataset creation, we utilize synthetic data generation.

reader = SimpleDirectoryReader(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/data/paul_graham")
documents = reader.load_data()


parser = SentenceSplitter(chunk_size=768, chunk_overlap=50)
nodes = parser(documents)

vector_index = VectorStoreIndex(nodes)

retriever = vector_index.as_retriever(similarity_top_k=2)

retrieved_nodes = retriever.retrieve(eval_query)


for node in retrieved_nodes:
    display_jet_source_node(node, source_length=2000)

qa_dataset = generate_question_context_pairs(
    nodes, llm=gpt4, num_questions_per_chunk=2
)

queries = qa_dataset.queries.values()
print(list(queries)[5])

len(list(queries))

retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=retriever
)

sample_id, sample_query = list(qa_dataset.queries.items())[0]
sample_expected = qa_dataset.relevant_docs[sample_id]

eval_result = retriever_evaluator.evaluate(sample_query, sample_expected)
print(eval_result)

eval_results = retriever_evaluator.evaluate_dataset(qa_dataset)


def display_results(name, eval_results):
    """Display results from evaluate."""

    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    hit_rate = full_df["hit_rate"].mean()
    mrr = full_df["mrr"].mean()

    metric_df = pd.DataFrame(
        {"retrievers": [name], "hit_rate": [hit_rate], "mrr": [mrr]}
    )

    return metric_df


display_results("top-2 eval", eval_results)

logger.info("\n\n[DONE]", bright=True)
