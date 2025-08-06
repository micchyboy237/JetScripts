import json
import argparse
import subprocess
import os
from jet.visualization.plot_topics import process_documents_for_chart


def main():
    parser = argparse.ArgumentParser(
        description="Generate topic visualization data and optionally create R plot.")
    parser.add_argument("--output-dir", default=".",
                        help="Directory to save category counts CSV and Chart.js JSON")
    parser.add_argument("--min-topic-size", type=int,
                        default=2, help="Minimum topic size for BERTopic")
    parser.add_argument("--run-r-plot", action="store_true",
                        help="Run R script to generate plot after processing")

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

    if args.run_r_plot:
        csv_path = os.path.join(args.output_dir, "category_counts.csv")
        plot_path = os.path.join(args.output_dir, "topics_plot.png")
        r_script = "plot_topics.R"
        if not os.path.exists(r_script):
            print(
                f"Error: R script '{r_script}' not found in current directory")
            return

        try:
            subprocess.run([
                "Rscript", "-e",
                f"source('{r_script}'); create_topic_bar_chart('{csv_path}', '{plot_path}')"
            ], check=True)
            print(f"Plot saved to {plot_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error running R script: {e}")


if __name__ == "__main__":
    main()
