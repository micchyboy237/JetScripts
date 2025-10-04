import json
import argparse
import subprocess
import os
from jet.libs.bertopic.jet_examples.base.plot_topics import process_documents_for_chart
from jet.logger import logger

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
R_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/visualization/charts/plot_topics.R"


def main():
    parser = argparse.ArgumentParser(
        description="Generate topic visualization data and optionally create R plot.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
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
    logger.debug(json.dumps(chart_config, indent=2))

    # Generate R plot
    csv_path = os.path.join(args.output_dir, "category_counts.csv")
    plot_path = os.path.join(args.output_dir, "topics_plot.png")

    # Ensure R script exists
    if not os.path.exists(R_PATH):
        raise FileNotFoundError(f"R script not found: {R_PATH}")

    try:
        logger.info("Generating R plot...")
        # Log CSV content for debugging
        csv_path = os.path.join(args.output_dir, "category_counts.csv")
        if os.path.exists(csv_path):
            with open(csv_path, "r") as f:
                logger.debug(f"CSV content:\n{f.read()}")
        else:
            logger.error(f"CSV file not found: {csv_path}")
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        result = subprocess.run([
            "Rscript", "-e",
            f"source('{R_PATH}'); create_topic_bar_chart('{csv_path}', '{plot_path}')"
        ], check=True, capture_output=True, text=True)
        logger.debug(f"R script output: {result.stdout}")
        if result.stderr:
            logger.debug(f"R script stderr: {result.stderr}")
        logger.success(f"Plot saved to {plot_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running R script: {e}")
        logger.error(f"R script stderr: {e.stderr}")
        raise
    except FileNotFoundError as e:
        logger.error(str(e))
        raise


if __name__ == "__main__":
    main()
