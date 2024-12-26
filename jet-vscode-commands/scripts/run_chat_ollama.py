import json
import sys
import time
from typing import Generator
from jet.llm import call_ollama_chat
from jet.logger import logger

DEFAULT_PROMPT = "Write a short filipino joke."
DEFAULT_MODEL = "llama3.1"

PROMPT_TEMPLATE = "Context information is below.\n---------------------\n{context_str}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {query_str}\nAnswer: "


def get_user_input():
    prompt = input("Enter the prompt: ") or DEFAULT_PROMPT
    logger.debug(prompt)
    models = ["llama3.1", "llama3.2", "codellama"] or DEFAULT_MODEL
    model_index = 0
    print("\nAvailable models:")
    for i, model in enumerate(models):
        print(f"{i+1}. {model}")
    while True:
        choice = input("Choose a model (enter the number): ")
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            model_index = int(choice) - 1
            break
        elif choice in models:
            model_index = models.index(choice)
            break
        else:
            print("Invalid choice. Please try again.")
    logger.debug(models[model_index])

    return prompt, models[model_index]


def handle_stream_response(stream_response: Generator[str, None, None]) -> str:
    output = ""
    for chunk in stream_response:
        output += chunk
    return output


def get_args():
    query = "Refactor the code"

    file_path = sys.argv[0]
    line_number = int(sys.argv[1]) if len(sys.argv) > 1 else None

    with open(file_path, 'r') as file:
        content = file.read()

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
    while True:
        print("\nOptions:")
        print("1. Send a message")
        print("2. Quit")
        choice = input("Enter the number of your choice: ") or "1"
        logger.debug(choice)
        if choice == "1":
            prompt, model = get_user_input()

            # Call the Ollama Chat API
            response = call_ollama_chat(
                prompt,
                model=model,
            )
            output = handle_stream_response(response)
        elif choice == "2":
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
