"""
RAG Pitfalls Demo: Evaluation & Monitoring
Anti-pattern: No Evaluation Dataset
Solution:     Golden dataset with automated context precision/recall scoring

Builds a small golden dataset and evaluates retrieval quality using
llama-index's built-in RetrieverEvaluator.

Reference: rag-pitfalls.md → Evaluation → No Evaluation Dataset
"""

from jet.adapters.llama_index.factory import (
    get_llama_cpp_embed_model,
    get_llama_cpp_llm,
)
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.evaluation import RetrieverEvaluator

Settings.llm = get_llama_cpp_llm()
Settings.embed_model = get_llama_cpp_embed_model()

documents = [
    Document(
        text="E-4092: Database connection timeout in payment service. Fix: increase pool_size.",
        metadata={"doc_id": "err-4092"},
    ),
    Document(
        text="E-5001: Invalid API key format. Must start with jk_live_ or jk_test_.",
        metadata={"doc_id": "err-5001"},
    ),
    Document(
        text="E-3010: Redis cache eviction exceeded. Set maxmemory-policy allkeys-lru.",
        metadata={"doc_id": "err-3010"},
    ),
    Document(
        text="Incident 2024-03-15: Payment degradation from AWS us-east-1 outage.",
        metadata={"doc_id": "inc-0315"},
    ),
    Document(
        text="SKU-A7X-200: Wireless charging pad, Qi2 compatible, $29.99.",
        metadata={"doc_id": "sku-a7x"},
    ),
]

vector_index = VectorStoreIndex.from_documents(documents)
retriever = vector_index.as_retriever(similarity_top_k=3)

# ── ✅ SOLUTION: Golden Dataset Evaluation ──────────────────────────────────
# Each entry: (question, expected_doc_ids)
golden_dataset = [
    {"query": "What does error E-4092 mean?", "expected_ids": ["err-4092"]},
    {"query": "How do I fix invalid API key errors?", "expected_ids": ["err-5001"]},
    {"query": "What happened on March 15 2024?", "expected_ids": ["inc-0315"]},
    {
        "query": "Tell me about the wireless charger product",
        "expected_ids": ["sku-a7x"],
    },
    {"query": "Redis memory eviction issues", "expected_ids": ["err-3010"]},
]

retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["hit_rate", "mrr"],
    retriever=retriever,
)

print("=" * 60)
print("✅ GOLDEN DATASET EVALUATION")
print("=" * 60)
print(f"  Evaluating {len(golden_dataset)} queries against retriever...\n")

results = []
for item in golden_dataset:
    # Official API uses keyword arguments [[27]]
    eval_result = retriever_evaluator.evaluate(
        query=item["query"],
        expected_ids=item["expected_ids"],
    )
    results.append(eval_result)
    hit = eval_result.metric_dict.get("hit_rate", 0)
    mrr = eval_result.metric_dict.get("mrr", 0)
    status = "✅" if hit == 1.0 else "❌"
    print(f"  {status} '{item['query'][:50]}...'")
    print(f"      Hit Rate: {hit:.2f} | MRR: {mrr:.2f}")

avg_hit = sum(r.metric_dict.get("hit_rate", 0) for r in results) / len(results)
avg_mrr = sum(r.metric_dict.get("mrr", 0) for r in results) / len(results)
print(f"\n  📊 AVERAGE — Hit Rate: {avg_hit:.2f} | MRR: {avg_mrr:.2f}")
