import sys
from jet.llm import call_ollama_chat
from jet.logger import logger

DEFAULT_MODEL = "llama3.1"

SYSTEM_MESSAGE = """
You are an AI assistant that follows instructions. 
You help refactor existing code, understand and write code of any programming language, extract code from unstructured web content, fix bugs and syntax errors, and write clean, optimized, readable, and modular code.
You provide real-world usage examples to demonstrate the features.
Output only the generated code without any explanations wrapped in a code block.
""".strip()

PROMPT_TEMPLATE = "Context information is below.\n---------------------\n{context_str}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {query_str}\nAnswer: "


def main():
    file_path = sys.argv[1]
    query = "Refactor the code"

    try:
        if len(sys.argv) == 1 or not sys.argv[1]:
            with open(file_path, 'r') as file:
                content = file.read()
        else:
            highlighted_text = sys.argv[1]
            content = highlighted_text

        prompt = PROMPT_TEMPLATE.format(
            context_str=content,
            query_str=query,
        )

        output = ""
        for chunk in call_ollama_chat(
            prompt,
            stream=True,
            model=DEFAULT_MODEL,
            system=SYSTEM_MESSAGE,
            options={
                "seed": 42,
                "num_ctx": 2048,
                "num_keep": 0,
                "num_predict": -1,
                "temperature": 0,
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

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
