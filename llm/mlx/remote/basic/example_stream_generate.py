from jet.llm.mlx.remote import generation as gen
from jet.logger import logger


def main():
    logger.info("=== Streaming Text Generation ===")
    for chunk in gen.stream_generate(
        "In the future, AI assistants will",
        model=None,
        max_tokens=100
    ):
        if "choices" in chunk and chunk["choices"]:
            token = chunk["choices"][0].get("text")
            if token:
                logger.teal(token, flush=True)
    logger.info("\n--- Stream End ---")


if __name__ == "__main__":
    main()
