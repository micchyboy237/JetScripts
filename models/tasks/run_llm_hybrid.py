from llama_cpp import Llama

from jet.models.tasks.llm_search import search_docs
from jet.models.tasks.llm_rerank import rerank_docs

if __name__ == "__main__":
    model_path = "/Users/jethroestrada/Downloads/Qwen3-Embedding-0.6B-f16.gguf"
    model = Llama(
        model_path=model_path,
        embedding=True,
        n_ctx=512,
        n_threads=8,
        n_gpu_layers=99,
        n_threads_batch=8,
        verbose=True
    )
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    queries = [
        'What is the capital of China?',
        # 'Explain gravity'
    ]
    documents = [
        "The capital of China is Beijing.",
        "China is a country in East Asia with a rich history.",
        "Gravity is a force that attracts two bodies towards each other."
    ]

    try:
        # Initial search
        search_results = search_docs(model, queries, documents, task)
        for query_idx, query_results in enumerate(search_results):
            print(f"\nQuery: {queries[query_idx]} (Search Results)")
            for res in query_results:
                print(
                    f"Rank: {res['rank']}, Score: {res['score']:.4f}, Text: {res['text']}")

        # Rerank results
        rerank_results = rerank_docs(model, queries, search_results)
        for query_idx, query_results in enumerate(rerank_results):
            print(f"\nQuery: {queries[query_idx]} (Reranked Results)")
            for res in query_results:
                print(
                    f"Rank: {res['rank']}, Score: {res['score']:.4f}, Text: {res['text']}")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        model.close()
