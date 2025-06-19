import os
from jet.file.utils import load_file, save_file
from jet.llm.evaluators.context_relevancy_evaluator import evaluate_context_relevancy
from jet.models.model_types import LLMModelType


def demo_low_relevancy(model: LLMModelType):
    """Demonstrates a low relevancy evaluation with unrelated context."""
    query = "What is the capital of France?"
    context = "The theory of relativity was developed by Albert Einstein."
    result = evaluate_context_relevancy(
        query=query,
        contexts=context,
        model=model
    )
    return result


def demo_medium_relevancy(model: LLMModelType):
    """Demonstrates a medium relevancy evaluation with partially related context."""
    query = "What is the capital of France?"
    context = "Paris hosts many tourists in France."
    result = evaluate_context_relevancy(
        query=query,
        contexts=context,
        model=model
    )
    return result


def demo_high_relevancy(model: LLMModelType):
    """Demonstrates a high relevancy evaluation with directly relevant context."""
    query = "What is the capital of France?"
    context = "The capital of France is Paris."
    result = evaluate_context_relevancy(
        query=query,
        contexts=context,
        model=model
    )
    return result


if __name__ == "__main__":
    # Original main execution
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/contexts.json"
    context_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_isekai_anime_2025/context.md"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    llm_model: LLMModelType = "qwen3-1.7b-4bit"
    docs = load_file(docs_file)
    context = load_file(context_file)
    query = docs["query"]
    result = evaluate_context_relevancy(
        query,
        context,
        llm_model,
    )
    save_file(result, f"{output_dir}/evaluation_result.json")

    # Run demonstration functions
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
