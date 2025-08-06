import json
import argparse
from jet.visualization.plot_topics import process_documents_for_chart


def main():
    parser = argparse.ArgumentParser(
        description="Generate topic visualization data from documents.")
    parser.add_argument("--output-dir", default=".",
                        help="Directory to save category counts CSV and Chart.js JSON")
    parser.add_argument("--min-topic-size", type=int,
                        default=2, help="Minimum topic size for BERTopic")

    args = parser.parse_args()

    documents = [
        {"id": 1, "content": "Advances in artificial intelligence are transforming industries."},
        {"id": 2, "content": "Stock market trends indicate a bullish economy."},
        {"id": 3, "content": "Machine learning models improve prediction accuracy."},
        {"id": 4, "content": "New vaccine developed for infectious disease."},
        {"id": 5, "content": "Neural networks are key to modern AI systems."},
        {"id": 6, "content": "Investment strategies for a volatile market."}
    ]

    chart_config = process_documents_for_chart(
        documents, args.output_dir, args.min_topic_size)
    print(json.dumps(chart_config, indent=2))


if __name__ == "__main__":
    main()
