import os
from jet.file.utils import load_file, save_file
from jet.llm.evaluators.response_relevancy_evaluator import evaluate_response_relevancy
from jet.models.model_types import LLMModelType


def demo_low_relevancy(model: LLMModelType):
    """Demonstrates a low relevancy evaluation with unrelated response."""
    query = "What is the capital of France?"
    response = "The theory of relativity was developed by Albert Einstein."
    result = evaluate_response_relevancy(
        query=query,
        response=response,
        model=model
    )
    return result


def demo_medium_relevancy(model: LLMModelType):
    """Demonstrates a medium relevancy evaluation with partially related response."""
    query = "What is the capital of France?"
    response = "Paris is a major city in France."
    result = evaluate_response_relevancy(
        query=query,
        response=response,
        model=model
    )
    return result


def demo_high_relevancy(model: LLMModelType):
    """Demonstrates a high relevancy evaluation with directly relevant response."""
    query = "What is the capital of France?"
    response = "The capital of France is Paris."
    result = evaluate_response_relevancy(
        query=query,
        response=response,
        model=model
    )
    return result


if __name__ == "__main__":
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/contexts.json"
    response_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/response.md"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    llm_model: LLMModelType = "qwen3-1.7b-4bit"
    docs = load_file(docs_file)
    response = load_file(response_file)
    query = docs["query"]
    result = evaluate_response_relevancy(
        query=query,
        response=response,
        model=llm_model,
    )
    save_file(result, f"{output_dir}/evaluation_result.json")
    print("Running low relevancy demo...")
    demo_low_relevancy_result = demo_low_relevancy(llm_model)
    save_file(demo_low_relevancy_result,
              f"{output_dir}/demo_low_relevancy.json")
    print("Running medium relevancy demo...")
    demo_medium_relevancy_result = demo_medium_relevancy(llm_model)
    save_file(demo_medium_relevancy_result,
              f"{output_dir}/demo_medium_relevancy.json")
    print("Running high relevancy demo...")
    demo_high_relevancy_result = demo_high_relevancy(llm_model)
    save_file(demo_high_relevancy_result,
              f"{output_dir}/demo_high_relevancy.json")
