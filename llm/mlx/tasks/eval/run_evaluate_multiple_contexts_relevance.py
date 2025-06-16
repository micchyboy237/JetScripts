import json
import os
import shutil

from jet.code.splitter_markdown_utils import get_header_level
from jet.data.stratified_sampler import ProcessedDataString, StratifiedSampler
from jet.file.utils import load_file, save_file
from jet.llm.mlx.tasks.eval.evaluate_multiple_contexts_relevance import (
    evaluate_multiple_contexts_relevance,
    load_model_components,
    load_classifier,
    save_classifier,
    ExtendedModelComponents,
    ContextRelevanceResult
)
from jet.vectors.document_types import HeaderDocument
from jet.wordnet.words import get_words

from collections import Counter, defaultdict
from statistics import mean


def summarize_results(results: list[ContextRelevanceResult]) -> dict:
    priority_counts = Counter(res["priority"]
                              for res in results if res["is_valid"])
    total_valid = sum(priority_counts.values())

    def percent(val: int) -> float:
        return (val / total_valid * 100) if total_valid else 0.0

    # Prepare probability groups by priority
    # {priority: [[p0s], [p1s], [p2s]]}
    prob_by_priority = defaultdict(lambda: [[] for _ in range(3)])
    for res in results:
        if res["is_valid"]:
            for i in range(3):
                prob_by_priority[res["priority"]][i].append(
                    res["probabilities"][i])

    # Compute mean probabilities per priority
    mean_probs_by_priority = {
        priority: [round(mean(prob_list), 4)
                   if prob_list else 0.0 for prob_list in prob_lists]
        for priority, prob_lists in prob_by_priority.items()
    }

    def prob_diff(p1: str, p2: str) -> list[float]:
        return [
            round(mean_probs_by_priority.get(p1, [0, 0, 0])[
                  i] - mean_probs_by_priority.get(p2, [0, 0, 0])[i], 4)
            for i in range(3)
        ]

    summary = {
        "total_valid": total_valid,
        "counts": dict(priority_counts),
        "percentages": {
            "high": round(percent(priority_counts.get("high", 0)), 2),
            "medium": round(percent(priority_counts.get("medium", 0)), 2),
            "low": round(percent(priority_counts.get("low", 0)), 2),
        },
        "differences": {
            "high_vs_medium": round(percent(priority_counts.get("high", 0)) - percent(priority_counts.get("medium", 0)), 2),
            "high_vs_low": round(percent(priority_counts.get("high", 0)) - percent(priority_counts.get("low", 0)), 2),
            "medium_vs_low": round(percent(priority_counts.get("medium", 0)) - percent(priority_counts.get("low", 0)), 2),
        },
        "probability_means": mean_probs_by_priority,
        "probability_differences": {
            "high_vs_medium": prob_diff("high", "medium"),
            "high_vs_low": prob_diff("high", "low"),
            "medium_vs_low": prob_diff("medium", "low"),
        }
    }

    return summary


def format_sub_dir(text: str) -> str:
    return text.lower().strip('.,!?').replace(' ', '_').replace(
        '.', '_').replace(',', '_').replace('!', '_').replace('?', '_').strip()


if __name__ == "__main__":
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    docs = load_file(docs_file)
    query = docs["query"]

    docs = [HeaderDocument(**doc) for doc in docs["documents"]]
    doc_texts = [
        f"{doc["header"].lstrip('#').strip()}\n{doc["content"]}" for doc in docs]

    contexts = doc_texts
    model_path = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"

    query_sub_dir = format_sub_dir(query)
    output_dir = os.path.join(output_dir, query_sub_dir)
    shutil.rmtree(output_dir, ignore_errors=True)

    save_dir = f"{os.path.dirname(__file__)}/saved_models/model_scraped_html"

    mock_data = load_file(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/mlx/tasks/data/inputs/query-context-relevance-labels.json")
    data: list[ProcessedDataString] = [
        ProcessedDataString(
            source="Query: {}\nContext: {}".format(
                d["query"], d["context"].lstrip('#').strip()),
            category_values=[
                d["label"],
                d["query"],
                get_header_level(d["context"]),
                *get_words(d["context"].splitlines()[0])
                # *get_words(d["context"])
            ]
        ) for d in mock_data
    ]
    sampler = StratifiedSampler(data)
    train_data, test_data, val_data = sampler.split_train_test_val(
        train_ratio=0.6, test_ratio=0.2)
    save_file(train_data, f"{output_dir}/dataset/train.json")
    save_file(test_data, f"{output_dir}/dataset/test.json")
    save_file(val_data, f"{output_dir}/dataset/val.json")

    default_pairs = [d["source"] for d in train_data]
    default_labels = [int(d["category_values"][0]) for d in train_data]
    save_file({
        "pairs": default_pairs,
        "labels": default_labels
    }, f"{output_dir}/formatted_train_data.json")

    # Training step
    classifier, label_encoder, embedder = load_classifier(
        save_dir, default_pairs, default_labels, verbose=True, overwrite=True)
    save_classifier(classifier, label_encoder,
                    embedder, save_dir, verbose=True)

    # Inference step
    model_components = load_model_components(model_path, verbose=True)
    extended_components = ExtendedModelComponents(
        model_components.model, model_components.tokenizer, classifier, label_encoder, embedder)
    results = evaluate_multiple_contexts_relevance(
        query, contexts, extended_components, verbose=True)

    print(f"Query: {query}")
    for res in results:
        print(f"Context: {json.dumps(res['context'])[:100]}")
        print(
            f"Relevance Score: {res['relevance_score']} (Score: {res['score']:.4f})")
        print(
            f"Probabilities (0, 1, 2): {[f'{p:.4f}' for p in res['probabilities']]}")
        print(f"Priority: {res['priority']}")
        print(f"Valid: {res['is_valid']}, Error: {res['error']}\n")

    save_file(results, f"{output_dir}/results.json")

    summary = summarize_results(results)
    save_file(summary, f"{output_dir}/summary.json")
