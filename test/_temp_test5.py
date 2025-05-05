import os
from jet.llm.mlx.templates.generate_tags import generate_tags
from jet.logger.logger import CustomLogger


# Setup logger
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")


def main():
    example_text = "Artificial intelligence is transforming healthcare and finance."

    result = generate_tags(example_text)

    if result:
        logger.info(f"Tags: {result['tags']}")
    else:
        logger.error("Failed to extract tags.")


if __name__ == "__main__":
    main()
