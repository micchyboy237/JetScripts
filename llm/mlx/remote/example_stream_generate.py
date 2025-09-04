from jet.llm.mlx.remote import generation as gen
from jet.logger import logger


def main():
    logger.info("=== Streaming Text Generation ===")
    for chunk in gen.stream_generate(
        "In the future, AI assistants will",
        model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    ):
        if "choices" in chunk and chunk["choices"]:
            token = chunk["choices"][0].get("text")
            if token:
                logger.teal(token, flush=True)
    logger.info("\n--- Stream End ---")


if __name__ == "__main__":
    main()
