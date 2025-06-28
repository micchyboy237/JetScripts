from jet.vectors.gguf.vector_search import VectorSearch, get_detailed_instruct


if __name__ == "__main__":
    # Model configuration
    model_path = "/Users/jethroestrada/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B-GGUF/snapshots/8aa0010e73a1075e99dfc213a475a60fd971bbe7/Qwen3-Embedding-0.6B-f16.gguf"
    task = 'Given a web search query, retrieve relevant passages that answer the query'

    # Input data (same as provided)
    queries = [
        get_detailed_instruct(task, 'What is the capital of China?'),
    ]
    documents = [
        "The capital of China is Beijing.",
        "China is a country in East Asia with a rich history.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
    ]

    try:
        # Initialize vector search
        searcher = VectorSearch(model_path, n_ctx=2048,
                                n_threads=4, n_gpu_layers=0)

        # Example 1: Single query (same as original)
        print("\nExample 1: Single query - Capital of China")
        query_embeddings = searcher.encode_texts(queries)
        doc_embeddings = searcher.encode_texts(documents)
        query_embeddings = searcher.normalize_embeddings(query_embeddings)
        doc_embeddings = searcher.normalize_embeddings(doc_embeddings)
        scores = searcher.compute_similarity_matrix(
            query_embeddings, doc_embeddings)
        print("Similarity matrix:")
        print(scores.tolist())

        # Example 2: Multiple queries
        print("\nExample 2: Multiple queries")
        queries_multi = [
            get_detailed_instruct(task, 'What is the capital of China?'),
            get_detailed_instruct(task, 'Explain gravity')
        ]
        query_embeddings = searcher.encode_texts(queries_multi)
        query_embeddings = searcher.normalize_embeddings(query_embeddings)
        scores = searcher.compute_similarity_matrix(
            query_embeddings, doc_embeddings)
        print("Similarity matrix (multiple queries):")
        print(scores.tolist())

        # Example 3: Empty query edge case
        print("\nExample 3: Empty query")
        queries_empty = [get_detailed_instruct(task, '')]
        query_embeddings = searcher.encode_texts(queries_empty)
        query_embeddings = searcher.normalize_embeddings(query_embeddings)
        scores = searcher.compute_similarity_matrix(
            query_embeddings, doc_embeddings)
        print("Similarity matrix (empty query):")
        print(scores.tolist())

        # Example 4: Long query
        print("\nExample 4: Long query")
        queries_long = [get_detailed_instruct(
            task, 'What is the historical significance of Beijing as the capital of China, and how has it influenced Chinese culture?')]
        query_embeddings = searcher.encode_texts(queries_long)
        query_embeddings = searcher.normalize_embeddings(query_embeddings)
        scores = searcher.compute_similarity_matrix(
            query_embeddings, doc_embeddings)
        print("Similarity matrix (long query):")
        print(scores.tolist())

    except Exception as e:
        print(f"Error during processing: {str(e)}")
