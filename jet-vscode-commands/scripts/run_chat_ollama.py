import json
import sys
import time
from typing import Generator
from pynput.keyboard import Listener
from jet.llm import call_ollama_chat
from jet.logger import logger

DEFAULT_QUERY = "Summarize provided context."
DEFAULT_MODEL = "llama3.1"

PROMPT_TEMPLATE = "Context information is below.\n---------------------\n{context}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {query}\nAnswer: "

# Function to handle key press for model selection


def on_press_wrapper(models: list[str], model_selected: list[str]):
    def on_press(key):
        try:
            # Check if the key is a number
            if hasattr(key, 'char') and key.char in ['1', '2', '3']:
                # Select model based on the key pressed
                selected_model = models[int(key.char) - 1]
                model_selected.append(selected_model)  # Store selected model
                print(f"Model {key.char} selected: {selected_model}")
                return False  # Stop the listener after selection
        except AttributeError as e:
            print(f"Error: {e}")
    return on_press


def get_user_input(context: str = "", models: list[str] = [DEFAULT_MODEL], template: str = PROMPT_TEMPLATE):
    query = input("Enter the prompt: ") or DEFAULT_QUERY
    prompt = template.format(
        context=context,
        query=query,
    )

    print("\nAvailable models:")
    for i, model in enumerate(models):
        logger.log(f"[{i+1}]", model, colors=["WHITE", "GRAY"])

    model_selected = []  # List to store the selected model
    # Listen for key press to select the model
    with Listener(on_press=on_press_wrapper(models, model_selected)) as listener:
        listener.join()

    if model_selected:
        selected_model = model_selected[0]  # Get the first selected model
    else:
        # Default to the first model if none is selected
        selected_model = models[0]

    return prompt, selected_model


def handle_stream_response(stream_response: Generator[str, None, None]) -> str:
    output = ""
    for chunk in stream_response:
        output += chunk
    return output


def get_args():
    file_path = sys.argv[0]
    line_number = int(sys.argv[1]) if len(sys.argv) > 1 else None
    selected_text = sys.argv[2] if len(sys.argv) > 2 else None

    if not selected_text:
        with open(file_path, 'r') as file:
            context = file.read()
    else:
        context = selected_text
        logger.info("SELECTED_TEXT")
        logger.debug(context)

    return {
        "context": context,
        "line_number": line_number,
    }


def main():
    args_dict = get_args()

    while True:
        logger.newline()

        context = args_dict["context"]
        logger.info("CONTEXT:")
        logger.debug(context)

        logger.newline()
        models = ["llama3.1", "llama3.2", "codellama"]

        logger.newline()
        prompt, model = get_user_input(context=context, models=models)

        # Call the Ollama Chat API
        response = call_ollama_chat(
            prompt,
            model=model,
            options={
                "seed": -1,
                # "temperature": 0,
                "num_keep": 0,
                "num_predict": -1,
            },
        )
        output = handle_stream_response(response)
        print(output)  # Output from the response


if __name__ == "__main__":
    main()
