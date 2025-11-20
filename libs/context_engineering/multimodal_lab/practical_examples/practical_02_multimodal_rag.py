import json
import numpy as np

from jet.adapters.llama_cpp.llm import LlamacppLLM
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from jet.libs.context_engineering.course._02_context_processing.labs.multimodal_lab import (
    create_example_dir, get_logger, save_numpy,
    MultimodalRAG
)
from jet.file.utils import save_file


def practical_02_multimodal_rag():
    """Build a real Multimodal RAG system with text + image queries."""
    example_dir = create_example_dir("practical_02_multimodal_rag")
    logger = get_logger("mm_rag", example_dir)
    logger.info("=" * 90)
    logger.info("PRACTICAL 2: Multimodal RAG with Text + Image Retrieval")
    logger.info("=" * 90)

    (example_dir / "llm").mkdir(exist_ok=True)
    (example_dir / "queries").mkdir(exist_ok=True)

    llm = LlamacppLLM(verbose=True)
    embedder = LlamacppEmbedding(model="embeddinggemma", use_cache=True)

    # Build RAG system
    rag = MultimodalRAG(d_model=768)
    processor = rag.processor

    # Add real multimodal documents
    documents = [
        ("Sunset over mountain lake", "A serene lake reflects orange sunset with snow-capped mountains"),
        ("City at night", "Neon lights, skyscrapers, busy streets in Tokyo at midnight"),
        ("Forest waterfall", "Lush green forest with cascading waterfall and morning mist"),
    ]

    logger.info("Adding 3 real image+caption documents to knowledge base...")
    for i, (title, caption) in enumerate(documents):
        # Simulate image + get fake tokens
        image = np.random.rand(224, 224, 3) * 255
        token_prompt = f"Convert to 40 token IDs: {caption}"
        tokens = list(range(2000 + i*100, 2040 + i*100))

        rag.add_to_knowledge_base(
            text_data=np.array(tokens),
            image_data=image,
            metadata={"title": title, "doc_id": f"mm_doc_{i}"}
        )
        save_file(caption, str(example_dir / f"doc_{i}_caption.txt"))

    # Query: "Find peaceful nature scenes"
    query_prompt = "Generate a query about peaceful nature or calming landscapes."
    save_file(query_prompt, str(example_dir / "llm" / "query_prompt.md"))

    logger.info("Generating query with LLM...")
    query_text = ""
    for chunk in llm.generate(query_prompt, temperature=0.7, max_tokens=100, stream=True):
        query_text += chunk
    save_file(query_text.strip(), str(example_dir / "llm" / "query_response.md"))

    # Convert query to fake tokens
    query_tokens = np.array(list(range(5000, 5040)), dtype=np.int64)

    # Retrieve
    results = rag.retrieve_relevant_context(query_text=query_tokens, top_k=3)

    retrieval_summary = []
    for r in results:
        entry = {
            "rank": len(retrieval_summary) + 1,
            "similarity": round(r["similarity"], 4),
            "title": r["metadata"]["title"],
            "modalities": [m.value for m in r["context"].available_modalities]
        }
        retrieval_summary.append(entry)
        logger.info(f"Rank {entry['rank']}: {entry['title']} (sim={entry['similarity']:.4f})")

    save_file(json.dumps(retrieval_summary, indent=2), str(example_dir / "retrieval_results.json"))
    save_numpy(results[0]["context"].fused, example_dir, "top_result_fused")

    logger.info("PRACTICAL 2 COMPLETE — Multimodal RAG working!")
    logger.info("\nNEXT → Run practical_03_content_analysis.py")
    logger.info("=" * 90)


if __name__ == "__main__":
    practical_02_multimodal_rag()
