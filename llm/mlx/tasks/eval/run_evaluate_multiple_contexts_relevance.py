import json
import os
from jet.file.utils import load_file, save_file
from jet.llm.mlx.tasks.eval.evaluate_multiple_contexts_relevance import (
    evaluate_multiple_contexts_relevance,
    load_model_components,
    load_classifier,
    save_classifier,
    ExtendedModelComponents
)
from jet.vectors.document_types import HeaderDocument

if __name__ == "__main__":
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/docs.json"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    docs = load_file(docs_file)
    docs = [HeaderDocument(**doc) for doc in docs]
    doc_texts = [
        f"Header: {doc["header"]}\nContent: {doc["content"]}" for doc in docs]

    query = "List all ongoing and upcoming isekai anime 2025."
    contexts = doc_texts

    model_path = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"
    save_dir = f"{os.path.dirname(__file__)}/saved_models/model_scraped_html"

    train_data = load_file(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/mlx/tasks/data/inputs/query-context-labels-pairs.json")

    default_pairs = [
        "Query: {}\nContext: {}".format(d["query"], d["context"])
        for d in train_data
    ]
    default_labels = [d["label"] for d in train_data]

    # Training step
    # Training step
    classifier, label_encoder, embedder = load_classifier(
        save_dir=save_dir, verbose=True, overwrite=True)
    save_classifier(classifier, label_encoder,
                    embedder, save_dir, verbose=True)

    # Inference step
    model_components = load_model_components(model_path, verbose=True)
    extended_components = ExtendedModelComponents(
        model_components.model, model_components.tokenizer, classifier, label_encoder, embedder)
    result = evaluate_multiple_contexts_relevance(
        query, contexts, extended_components, verbose=True)

    print(f"Query: {result['query']}")
    for res in result['results']:
        print(f"Context: {json.dumps(res['context'])[:100]}")
        print(
            f"Relevance Score: {res['relevance_score']} (Confidence: {res['confidence']:.4f})")
        print(f"Priority: {res['priority']}")
        print(f"Valid: {res['is_valid']}, Error: {res['error']}\n")

    save_file(result, f"{output_dir}/result.json")
