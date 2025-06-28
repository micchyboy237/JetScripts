import os
from typing import List

from jet.code.markdown_utils import parse_markdown
from jet.file.utils import load_file, save_file
from jet.models.embeddings.sentence_transformer_pooling import PoolingMode, load_sentence_transformer, search_docs


if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    html_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_rag_strategies_reddit_2025/pages/www.reddit.com_r_rag_comments_1j4r4wj_10_rag_papers_you_should_read_from_february_2025/page.html"
    docs_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/features/generated/run_search_and_rerank/top_rag_strategies_reddit_2025/pages/www.reddit.com_r_rag_comments_1j4r4wj_10_rag_papers_you_should_read_from_february_2025/docs.json"

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    top_k = 10

    md_contents = parse_markdown(html_file)
    docs = load_file(docs_file)
    query = docs["query"]
    documents = [doc["content"] for doc in md_contents]

    # query = "How can I use data analytics to improve my ecommerce marketing strategy?"
    # documents = [
    #     "Our AI-powered analytics platform helps you understand customer behavior at scale, offering dashboards, anomaly detection, and churn prediction.",
    #     "This device complies with part 15 of the FCC Rules. Operation is subject to the following two conditions...",
    #     "A long time ago in a galaxy far, far away, a young farm boy discovers his destiny amidst intergalactic war and rebellion.",
    #     "At our law firm, we specialize in cross-border intellectual property litigation, patent prosecution, and trademark enforcement.",
    #     "We help ecommerce brands scale by building omnichannel marketing funnels using automation, audience segmentation, and real-time analytics.",
    #     "The software supports event-driven microservice architecture, written in TypeScript, deployable on AWS Lambda with DynamoDB and SQS integration."
    # ]

    pooling_strategies: List[PoolingMode] = [
        "cls_token",
        "mean_tokens",
        "max_tokens",
        "mean_sqrt_len_tokens"
    ]

    for strategy in pooling_strategies:
        print("=" * 80)
        model = load_sentence_transformer(model_name, pooling_mode=strategy)
        results = search_docs(model, documents, query, top_k=top_k)

        print(f"🔍 Pooling strategy: {strategy}")
        print(f"📌 Top results for query:\n'{query}'\n")
        for result in results:
            print(f"[{result['rank']}] Score: {result['score']:.4f}")
            print(f"{result['text']}\n")

        save_file({
            "model": model_name,
            "strategy": strategy,
            "results": results
        }, f"{output_dir}/{strategy}_results.json")
