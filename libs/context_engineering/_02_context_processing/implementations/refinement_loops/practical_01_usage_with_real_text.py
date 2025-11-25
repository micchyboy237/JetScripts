# examples/usage_with_real_text.py
import numpy as np
from typing import List
import hashlib

# --- 1. Use a modern, fast, free embedding model (Sentence Transformers) ---
from sentence_transformers import SentenceTransformer

# --- 2. Import the refinement system ---
from jet.libs.context_engineering.course._02_context_processing.implementations.refinement_loops import IterativeRefiner, RefinementPipeline, create_quality_report

# --- 3. Real-world example: RAG context refinement ---
def refine_context_for_query(
    query: str,
    documents: List[str],
    use_pipeline: bool = True,
    target_quality: float = 0.82,
    top_k: int = 20
) -> dict:
    """
    Full end-to-end example:
    - Embed query + documents
    - Stack top-k retrieved chunks as context
    - Run refinement loop
    - Return improved embeddings + quality report
    """
    if not documents:
        raise ValueError("No documents provided for refinement.")

    print(f"Query: {query!r}")
    print(f"Retrieved {len(documents)} document chunks (showing top {min(5, len(documents))})")
    for i, doc in enumerate(documents[:5], 1):
        print(f"   {i}. {doc.strip()[:120]}{'...' if len(doc) > 120 else ''}")

    # Load a strong, free embedding model (runs great on Mac M1 or Windows)
    print("\nLoading embedding model...")
    
    # Auto-detect device: MPS on Mac M1 (if available), else CPU
    import torch
    device_str = "mps" if torch.backends.mps.is_available() else "cpu"
    embedder = SentenceTransformer(
        "all-MiniLM-L6-v2",  # fast, 384-dim, excellent quality
        device=device_str
    )
    d_model = embedder.get_sentence_embedding_dimension()  # 384

    # Embed everything
    print("Encoding query and documents...")
    query_emb = embedder.encode(query, normalize_embeddings=True).reshape(1, -1)
    doc_embs = embedder.encode(documents, normalize_embeddings=True, show_progress_bar=True)

    # Simple BM25-style retrieval simulation: cosine similarity ranking
    similarities = doc_embs @ query_emb.T
    similarities = similarities.flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    context_embs = doc_embs[top_indices]  # shape: (top_k, d_model)

    print(f"Selected top {len(context_embs)} chunks for context")

    # --- Choose refiner: lightweight vs full pipeline ---
    if use_pipeline:
        refiner = RefinementPipeline(d_model=d_model, enable_caching=True)
        result = refiner.refine(
            context=context_embs,
            query=query_emb,
            target_quality=target_quality,
            constitutional_check=True
        )
        refined_context = None  # Pipeline doesn't return embeddings directly
        strategy_used = result["strategy_used"]
    else:
        refiner = IterativeRefiner(d_model=d_model, max_iterations=6, target_quality=target_quality)
        refined_context, result = refiner.refine(context_embs, query=query_emb)
        strategy_used = "adaptive"

    # Quality report
    print("\n" + "="*60)
    print("QUALITY IMPROVEMENT REPORT")
    print("="*60)
    if "initial_quality" in result:
        print(create_quality_report(result["initial_quality"]))
    print(create_quality_report(result["final_quality"]))

    print(f"Strategy used      : {strategy_used}")
    print(f"Iterations         : {result.get('iterations', 'N/A')}")
    print(f"Total time         : {result['total_processing_time']*1000:.1f} ms")
    print(f"Improvement        : {result['total_improvement']:+.4f}")
    print(f"Target reached     : {'Yes' if result['target_reached'] else 'No'}")

    return {
        "refined_embeddings": refined_context,  # None if using Pipeline
        "original_embeddings": context_embs,
        "query_embedding": query_emb,
        "selected_indices": top_indices.tolist(),
        "stats": result
    }


# --- Run the example ---
if __name__ == "__main__":
    query = "How does photosynthesis work in C4 plants compared to C3 plants?"

    documents = [
        "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water.",
        "In C3 plants, the first product of carbon fixation is 3-phosphoglycerate, a three-carbon compound.",
        "C4 plants have a special mechanism to concentrate CO2 around Rubisco, reducing photorespiration.",
        "The C4 pathway involves bundle sheath cells and mesophyll cells working together.",
        "Kranz anatomy is characteristic of C4 plants, with distinct bundle sheath and mesophyll cell layers.",
        "Photorespiration occurs when Rubisco binds oxygen instead of CO2, wasting energy.",
        "C4 plants are more efficient in hot, dry climates due to reduced photorespiration.",
        "Examples of C4 plants include maize, sugarcane, and sorghum.",
        "The Calvin cycle occurs in bundle sheath cells of C4 plants, isolated from atmospheric oxygen.",
        "PEP carboxylase in C4 plants has higher affinity for CO2 than Rubisco.",
        # ... imagine 100+ more chunks from a real vector DB
    ] * 15  # simulate larger retrieval set

    result = refine_context_for_query(
        query=query,
        documents=documents,
        use_pipeline=True,
        target_quality=0.82,
        top_k=16
    )