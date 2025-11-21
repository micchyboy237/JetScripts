import os
import numpy as np
from typing import Any, Dict, List
from jet.libs.context_engineering.course._02_context_processing.labs.self_refinement_lab import (
    create_example_dir, get_logger, SelfRefinementPipeline
)
from jet.file.utils import save_file

"""
PRACTICAL 8: Full local agent loop
Turn-by-turn polishing (like practical_04) + final answer from LLM
"""
def practical_08_live_dialogue_with_llm_response(
    dialogue_turns: List[str],
    llm: LlamacppLLM,
    embedder: LlamacppEmbedding,
) -> Dict[str, Any]:
    example_dir = create_example_dir("practical_05_llm_critique_refine")
    logger = get_logger("llm_critic", example_dir)

    save_file(dialogue_turns, os.path.join(example_dir, "dialogue_turns.json"))

    pipeline = SelfRefinementPipeline(max_iterations=1, d_model=768)
    current_context = np.zeros((0, 768))
    history = []

    for i, turn in enumerate(dialogue_turns):
        turn_emb = embedder.encode([turn], return_format="numpy")
        current_context = np.vstack([current_context, turn_emb]) if current_context.size else turn_emb

        if i % 2 == 1:  # Assistant turn → refine + generate response
            refined = pipeline.refine_context(current_context, turn_emb, target_quality=0.95)
            current_context = refined["final_context"]

            context_text = "\n".join(dialogue_turns[:i+1])
            response = llm.chat([{
                "role": "system",
                "content": "You are a helpful assistant. Use the refined context."
            }, {
                "role": "user",
                "content": context_text
            }], temperature=0.7)

            history.append({"user": turn, "assistant": response, "quality": refined["final_quality"].overall})

            save_file(history, os.path.join(example_dir, "history.json"), verbose=not history)

    return {"dialogue_history": history, "final_context": current_context}

if __name__ == "__main__":
    from jet.adapters.llama_cpp.llm import LlamacppLLM
    from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding

    llm = LlamacppLLM(model="qwen3-instruct-2507:4b")
    embedder = LlamacppEmbedding(model="embeddinggemma")

    dialogue = [
        "Hi, what is context engineering?",
        "It's about building high-quality prompts using retrieval, compression, and refinement.",
        "Can you give a real example?",
        "Sure. Self-refinement means the model critiques and improves its own context before answering.",
        "That's cool. Does it work in real-time chat?",
        "Yes! We can refine context every turn and then generate a better response.",
        "Show me how it improves quality over time.",
    ]

    result = practical_08_live_dialogue_with_llm_response(
        dialogue_turns=dialogue,
        llm=llm,
        embedder=embedder,
    )

    print("\nFULL DIALOGUE WITH PROGRESSIVE REFINEMENT:")
    for turn in result["dialogue_history"]:
        print(f"User: {turn['user']}")
        print(f"Assistant: {turn['assistant']}")
        print(f"   ↳ Context quality: {turn['quality']:.4f}\n")
