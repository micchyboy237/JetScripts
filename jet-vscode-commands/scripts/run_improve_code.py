import sys
from jet.actions import call_ollama_chat
from jet.logger import logger

DEFAULT_MODEL = "llama3.1"

SYSTEM_MESSAGE = """
You are an AI assistant that follows instructions. 
You help refactor existing code, understand and write code of any programming language, extract code from unstructured web content, fix bugs and syntax errors, and write clean, optimized, readable, and modular code.
You provide real-world usage examples to demonstrate the features.
Output only the generated code without any explanations wrapped in a code block.
""".strip()

PROMPT_TEMPLATE = "Context information is below.\n---------------------\n{context_str}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {query_str}\nAnswer: "


def get_args():
    query = "Refactor the code without breaking anything."

    file_path = sys.argv[0]
    line_number = int(sys.argv[1]) if len(sys.argv) > 1 else None
    selected_text = sys.argv[2] if len(sys.argv) > 2 else None

    if not selected_text:
        with open(file_path, 'r') as file:
            content = file.read()
    else:
        content = selected_text
        logger.info("SELECTED_TEXT")
        logger.debug(content)

    return {
        "query": query,
        "content": content,
        "line_number": line_number,
    }


def main():
    args_dict = get_args()

    prompt = PROMPT_TEMPLATE.format(
        context_str=args_dict["content"],
        query_str=args_dict["query"],
    )

    output = ""
    for chunk in call_ollama_chat(
        prompt,
        stream=True,
        model=DEFAULT_MODEL,
        system=SYSTEM_MESSAGE,
        options={
            "seed": 0,
            "temperature": 0,
            "num_keep": 0,
            "num_predict": -1,
        },
        # track={
        #     "repo": "./aim-logs",
        #     "experiment": "Code Enhancer Test",
        #     "run_name": "Improve python",
        #     "format": FINAL_MARKDOWN_TEMPLATE,
        #     "metadata": {
        #         "type": "code_enhancer",
        #     }
        # }
    ):
        output += chunk
    return output


if __name__ == "__main__":
    main()
