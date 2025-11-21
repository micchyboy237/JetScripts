import os
from typing import List
from jet.libs.context_engineering.course._02_context_processing.labs.self_refinement_lab import (
    create_example_dir, get_logger, ProductionRefinementSystem
)
from jet.file.utils import save_file

"""
PRACTICAL 9: LLM + Tools → reject bad context before expensive tool calls
"""
def search_tool(query: str) -> str:
    return "Mock search results about " + query

def practical_09_tool_use_quality_gate(
    query: str,
    context_chunks: List[str],
    llm: LlamacppLLM,
    embedder: LlamacppEmbedding,
) -> str:
    example_dir = create_example_dir("practical_05_llm_critique_refine")
    logger = get_logger("llm_critic", example_dir)

    save_file(query, os.path.join(example_dir, "query.md"))
    save_file(context_chunks, os.path.join(example_dir, "context_chunks.json"))

    # First: fast embedding gate
    system = ProductionRefinementSystem(d_model=768)
    emb = embedder.encode([query] + context_chunks, return_format="numpy")
    result = system.refine_context_production(emb[1:], emb[0:1])

    save_file(result, os.path.join(example_dir, "result.json"))

    if result["final_quality"].overall < 0.75:
        return "Context quality too low. Refusing to call tools."

    tools = [{
        "type": "function",
        "function": {
            "name": "search_tool",
            "description": "Search knowledge base",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }
    }]

    # Only if passes → allow tool use
    response = llm.chat_with_tools(
        messages=[{"role": "user", "content": query}],
        tools=tools,
        available_functions={"search_tool": search_tool},
        temperature=0.3,
    )
    return response

if __name__ == "__main__":
    import random
    from jet.adapters.llama_cpp.llm import LlamacppLLM
    from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding

    def real_search_tool(query: str) -> str:
        knowledge = {
            "context engineering": "Context engineering improves prompt quality through retrieval, compression, refinement, and self-correction.",
            "self-refinement": "Self-refinement allows LLMs to critique and iteratively improve their own context or answers.",
            "llama.cpp": "llama.cpp is a high-performance inference engine for running LLMs locally with quantization.",
        }
        return knowledge.get(query.lower(), f"Found 5 documents about '{query}' (simulated)")

    llm = LlamacppLLM(model="qwen3-instruct-2507:4b", base_url="http://shawn-pc.local:8080/v1")
    embedder = LlamacppEmbedding(model="embeddinggemma")

    # Test with intentionally bad context
    bad_chunks = [
        "The sky is green on Tuesdays.",
        "Context engineering is unrelated to AI.",
        "Pineapples grow on trees.",
    ]

    response = practical_09_tool_use_quality_gate(
        query="What is context engineering?",
        context_chunks=bad_chunks,
        llm=llm,
        embedder=embedder,
    )

    print("Response (with quality gate):")
    print(response)  # Expected: "Context quality too low. Refusing to call tools."
