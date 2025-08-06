import json
import argparse
import subprocess
import os
from jet.logger import logger
from jet.visualization.plot_topics import process_documents_for_chart

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])


def write_r_script(r_script_path: str) -> None:
    """Write the R script for creating a topic bar chart if it doesn't exist."""
    r_script_content = """
# Install required packages if not already installed
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2")
}
if (!requireNamespace("scales", quietly = TRUE)) {
  install.packages("scales")
}
if (!requireNamespace("stringr", quietly = TRUE)) {
  install.packages("stringr")
}

library(ggplot2)
library(scales)
library(stringr)

create_topic_bar_chart <- function(csv_path, output_path) {
  # Read the CSV file
  if (!file.exists(csv_path)) {
    stop("CSV file does not exist: ", csv_path)
  }
  
  data <- read.csv(csv_path, stringsAsFactors = FALSE)
  
  # Debug: Log CSV content
  cat("CSV content:\n")
  print(data)
  
  # Validate data
  if (!all(c("Topic", "Count") %in% colnames(data))) {
    stop("CSV must contain 'Topic' and 'Count' columns")
  }
  if (nrow(data) == 0) {
    stop("CSV file is empty")
  }
  if (nrow(data) == 1) {
    warning("Only one topic found; plot will show a single bar")
  }
  
  # Clean and shorten topic names
  data$Topic <- str_trunc(data$Topic, 20, "right")
  data$Topic <- str_replace_all(data$Topic, "[^[:alnum:]]", "_")
  
  # Define colors for bars
  colors <- c("steelblue", "darkorange", "purple", "darkred", "forestgreen")
  data$Color <- colors[seq_len(nrow(data)) %% length(colors) + 1]
  
  # Create bar chart
  p <- ggplot(data, aes(x = reorder(Topic, -Count), y = Count, fill = Color)) +
    geom_bar(stat = "identity") +
    scale_fill_identity() +
    theme_minimal() +
    labs(title = "Document Distribution by Topic",
         x = "Topic",
         y = "Number of Documents") +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 12, face = "bold"),
      axis.text.y = element_text(size = 12),
      axis.title = element_text(size = 14),
      plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
      panel.grid.major.x = element_blank()
    ) +
    scale_y_continuous(breaks = pretty_breaks(), expand = c(0, 0.5))
  
  # Save plot with higher resolution
  ggsave(output_path, plot = p, width = 10, height = 6, dpi = 600)
  cat("Plot saved to ", output_path, "\n")
}
"""
    os.makedirs(os.path.dirname(r_script_path), exist_ok=True)
    with open(r_script_path, "w") as f:
        f.write(r_script_content)
    logger.info(f"R script created at {r_script_path}")


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
    r_script = os.path.join(args.output_dir, "plot_topics.R")

    # Ensure R script exists
    if not os.path.exists(r_script):
        write_r_script(r_script)

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

        # Ensure R script exists
        r_script = os.path.join(args.output_dir, "plot_topics.R")
        if not os.path.exists(r_script):
            write_r_script(r_script)

        result = subprocess.run([
            "Rscript", "-e",
            f"source('{r_script}'); create_topic_bar_chart('{csv_path}', '{plot_path}')"
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
