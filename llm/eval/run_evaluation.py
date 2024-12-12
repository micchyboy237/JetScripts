from llama_index.core.llms import LLM
from llama_index.core.evaluation import FaithfulnessEvaluator, RetrieverEvaluator
from jet.vectors import SettingsManager, IndexManager
from jet.logger import logger


class EvaluationResult:
    """Class to hold evaluation results."""

    def __init__(self, passing: bool):
        self.passing = passing


def evaluate_response(llm: LLM, response: dict) -> EvaluationResult:
    """
    Evaluate a single response for faithfulness.

    Args:
    llm (LLM): The LLM instance.
    response (dict): The response to be evaluated.

    Returns:
    EvaluationResult: An object containing the evaluation result.
    """

    evaluator = FaithfulnessEvaluator(llm=llm)
    eval_result = evaluator.evaluate_response(response=response)

    for source_node in response.source_nodes:
        eval_result = evaluator.evaluate(
            response=response_str, contexts=[source_node.get_content()]
        )
        print(str(eval_result.passing))
    return eval_result


def evaluate_retrieval(retriever_evaluator: RetrieverEvaluator, query: str, expected_ids: list[str]) -> None:
    """
    Evaluate a single retrieval.

    Args:
    retriever_evaluator (RetrieverEvaluator): The retriever evaluator instance.
    query (str): The query to be evaluated.
    expected_ids (list[str]): A list of expected node IDs.
    """

    retriever_evaluator.evaluate(query=query, expected_ids=expected_ids)


def main() -> None:
    settings_manager = SettingsManager.create()
    # Create an OpenAI LLM instance
    llm = settings_manager.llm

    logger.debug("Creating nodes...")
    all_nodes = IndexManager.create_nodes(
        documents=documents, parser=settings_manager.node_parser)

    logger.debug("Creating index...")
    vector_index = IndexManager.create_index(
        embed_model=settings_manager.embed_model,
        nodes=all_nodes,
    )

    # Define a retriever evaluator
    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        ["mrr", "hit_rate"], retriever=vector_index.as_retriever(similarity_top_k=2)
    )

    # Evaluate a single response for faithfulness
    response = {"source": "context", "response": "answer"}
    eval_result = evaluate_response(llm, response)
    print(f"Faithfulness evaluation result: {eval_result.passing}")

    # Evaluate a single retrieval
    query = "query"
    expected_ids = ["node_id1", "node_id2"]
    evaluate_retrieval(retriever_evaluator, query, expected_ids)


if __name__ == "__main__":
    main()
