# JetScripts/libs/context_engineering/_02_context_processing/self_refinement_lab/practical_examples/practical_09_tool_use_quality_gate.py
"""
PRACTICAL 9: Quality Gate + Tool Guardrail
Perfect production pattern: try fast fix → fail fast → protect downstream
"""
from typing import List
from pathlib import Path

from jet.adapters.llama_cpp.llm import LlamacppLLM
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from jet.libs.context_engineering.course._02_context_processing.labs.self_refinement_lab import (
    create_example_dir,
    get_logger,
    ProductionRefinementSystem,
)
from jet.file.utils import save_file


def search_tool(query: str) -> str:
    return f"[MOCK SEARCH] Results for: {query}"


def practical_09_tool_use_quality_gate(
    query: str,
    context_chunks: List[str],
    llm: LlamacppLLM,
    embedder: LlamacppEmbedding,
    quality_threshold: float = 0.75,
) -> str:
    example_dir = create_example_dir("practical_09_tool_use_quality_gate")
    logger = get_logger("quality_gate", example_dir)
    logger.info("PRACTICAL 9: Quality Gate + Tool Guardrail")

    # Save inputs
    save_file(query, example_dir / "query.txt")
    save_file(context_chunks, example_dir / "context_chunks.json")

    # Embed
    logger.info(f"Embedding query + {len(context_chunks)} chunks")
    embeddings = embedder.encode([query] + context_chunks, return_format="numpy")
    query_emb, context_emb = embeddings[0:1], embeddings[1:]

    # Try to refine
    system = ProductionRefinementSystem(d_model=768)
    result = system.refine_context_production(context=context_emb, query=query_emb)

    init_q = result["initial_quality"].overall
    final_q = result["final_quality"].overall
    delta = result["total_improvement"]
    iters = result["iterations_completed"]

    # Save report
    report = {
        "initial_quality": init_q,
        "final_quality": final_q,
        "improvement": delta,
        "iterations": iters,
        "degradation_detected": delta < -0.01,
        "early_stop": iters < 3,
        "gate_passed": final_q >= quality_threshold,
    }
    save_file(report, example_dir / "quality_report.json")
    save_file(result, example_dir / "refinement_trace.json")

    # === EXPLAIN WHAT HAPPENED (this is the key section) ===
    logger.info(f"Quality: {init_q:.4f} → {final_q:.4f} (Δ={delta:+.4f}) after {iters} iter(s)")

    if delta < -0.01:
        logger.warning("Refinement made context WORSE → reverted (defensive guard worked)")
    elif delta == 0.0 and iters > 0:
        logger.warning(
            "Refinement had no effect → context likely unsalvageable (pure noise/off-topic)"
        )
    elif delta > 0.02:
        logger.info("Refinement succeeded — quality improved significantly")
    elif delta >= 0:
        logger.info("Minor or no improvement — but no harm done")

    # === FINAL GATE DECISION ===
    if final_q < quality_threshold:
        logger.warning(f"Quality {final_q:.4f} < {quality_threshold} → BLOCKING tool calls")
        return "Context quality too low. Refusing to call tools."

    logger.info(f"Quality {final_q:.4f} ≥ {quality_threshold} → allowing tool use")

    tools = [{
        "type": "function",
        "function": {
            "name": "search_tool",
            "description": "Search knowledge base",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    }]

    response = llm.chat_with_tools(
        messages=[{"role": "user", "content": query}],
        tools=tools,
        available_functions={"search_tool": search_tool},
        temperature=0.3,
    )

    save_file(response, example_dir / "tool_response.md")
    logger.info("Tool call executed successfully")
    return response


if __name__ == "__main__":
    llm = LlamacppLLM(model="qwen3-instruct-2507:4b", base_url="http://shawn-pc.local:8080/v1")
    embedder = LlamacppEmbedding(model="embeddinggemma")

    print("\n" + "="*80)
    print("PRACTICAL 9 — ALL POSSIBLE SCENARIOS")
    print("="*80)

    test_cases = [
        {
            "name": "Pure garbage (should BLOCK)",
            "query": "What is context engineering?",
            "chunks": [
                "The sky is green on Tuesdays.",
                "Context engineering is unrelated to AI.",
                "Pineapples grow on trees.",
                "Water is dry when you think about it.",
            ],
            "expected": "BLOCKED"
        },
        {
            "name": "Mostly good + one bad (should FIX + PASS)",
            "query": "What is context engineering?",
            "chunks": [
                "Context engineering improves prompt quality through retrieval, compression, and refinement.",
                "Self-refinement allows models to critique their own context.",
                "It reduces hallucinations and improves reasoning.",
                "The moon is made of cheese.",  # one bad chunk
            ],
            "expected": "PASSED"
        },
        {
            "name": "Perfectly clean (should PASS fast)",
            "query": "Explain self-refinement in RAG systems",
            "chunks": [
                "Self-refinement is an iterative process where the model critiques and improves its own context.",
                "It uses embedding-based quality assessment to detect issues.",
                "Common strategies include coherence smoothing and relevance boosting.",
                "Production systems use meta-controllers to adapt strategy per query.",
            ],
            "expected": "PASSED"
        },
        {
            "name": "Borderline — barely fixable (should PASS after effort)",
            "query": "How does quality gating work?",
            "chunks": [
                "Quality gating blocks low-scoring context from reaching the LLM.",
                "It's fast and runs before expensive operations.",
                "Sometimes it combines embedding and LLM assessors.",
                "I like pizza.",
                "Threshold is usually 0.7–0.8.",
            ],
            "expected": "PASSED"
        },
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {case['name']} ---")
        response = practical_09_tool_use_quality_gate(
            query=case["query"],
            context_chunks=case["chunks"],
            llm=llm,
            embedder=embedder,
            quality_threshold=0.75,
        )
        print(f"→ Decision: {response.split('.')[0].upper()}")
        print(f"   Expected: {case['expected']}")