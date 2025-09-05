import os
import shutil
from jet.llm.mlx.chat_history import ChatHistory
from jet.llm.mlx.remote import generation as gen
from jet.transformers.formatters import format_json
from jet.logger import logger
from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def main():
    print("\n=== Chat with Conversation History ===")
    history = ChatHistory()
    response1 = gen.chat(
        "Hello, who are you?",
        with_history=True,
        history=history,
        max_tokens=50,
        verbose=True,
    )
    logger.debug("Assistant:")
    logger.success(format_json(response1))
    save_file(response1, f"{OUTPUT_DIR}/response1.json")

    response2 = gen.chat(
        "Can you remind me what I just asked?",
        with_history=True,
        history=history,
        max_tokens=50,
        verbose=True,
    )
    logger.success(format_json(response2))
    save_file(response2, f"{OUTPUT_DIR}/response2.json")


if __name__ == "__main__":
    main()
