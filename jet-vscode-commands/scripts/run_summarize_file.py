import os
import random
import sys
import time
import json
from threading import Thread
from typing import Generator, List
from pynput import keyboard
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from collections import deque

from jet.llm import call_ollama_chat
from jet.utils import colorize_log, COLORS
from jet.logger import logger

DEFAULT_QUERY = "Summarize provided context."
DEFAULT_MODEL_SELECTION_KEYBOARD = {
    "cmd+1": "llama3.1",
    "cmd+2": "llama3.2",
    "cmd+3": "codellama",
}
DEFAULT_MODEL = "llama3.1"


PROMPT_TEMPLATE = """\
Context information is below.
---------------------
# File name: {file_name}

{context}
---------------------
Given the context information and not prior knowledge, answer the query.
Query:
{query}

Answer:
"""

FILE_NAME = os.path.basename(__file__)
SUMMARY_QUERY = "Your task is to generate a summary index that describes the general purpose the provided context, summarizing its real-world use cases in an easy-to-read format for other LLMs. Please describe the typical scenarios where this would be utilized, along with the benefits it brings to users. Keep it short and concise. Output only the generated answer without any explanations wrapped in a code block (use ```markdown)."


class HotReloadHandler(FileSystemEventHandler):
    def __init__(self, script_path: str):
        self.script_path = script_path

    def on_modified(self, event):
        if event.src_path == self.script_path:
            logger.newline()
            logger.info("File changed, restarting...")
            os.execv(sys.executable, [sys.executable] + sys.argv)


class ModelHandler:
    def __init__(
        self,
        default_model: str = DEFAULT_MODEL,
        model_selection_commands: dict[str,
                                       str] = DEFAULT_MODEL_SELECTION_KEYBOARD,
    ):
        self.default_model = default_model
        self.model_selection_commands = model_selection_commands
        self.commands = list(model_selection_commands.keys())
        self.models = list(model_selection_commands.values())
        self.selected_model = default_model
        self.queus_deque = deque(maxlen=5)
        self.listener_thread = Thread(
            target=self._start_key_listener, daemon=True)
        self.listener_thread.start()

        self._keys_pressed = []

    def _start_key_listener(self):
        def on_press(key):
            # Strip "Key." and surrounding "'" if exists
            updated_key = str(key).lstrip("Key.").strip("'")
            self._keys_pressed.append(updated_key)

        def on_release(key):
            command = "+".join(self._keys_pressed)

            if command in self.commands:
                self.selected_model = self.model_selection_commands[command]
                logger.log('Selected model:', self.selected_model,
                           colors=["DEBUG", "SUCCESS"])

            try:
                self._keys_pressed.pop()
            except IndexError as e:
                logger.error(e)

        with keyboard.Listener(
                on_press=on_press,
                on_release=on_release) as listener:
            listener.join()

    def get_user_input(self, context: str = "", template: str = PROMPT_TEMPLATE):
        query = SUMMARY_QUERY

        if not query:
            if self.queus_deque:
                query = self.queus_deque[-1]
            else:
                query = DEFAULT_QUERY

        # Update query memory queue
        if query in self.queus_deque:
            self.queus_deque.remove(query)
        self.queus_deque.append(query)

        logger.debug(query)

        template_args = {"context": context, "query": query}
        if "{file_name}" in template:
            template_args["file_name"] = FILE_NAME

        prompt = template.format(**template_args)

        return prompt, self.selected_model

    @staticmethod
    def handle_stream_response(stream_response: Generator[str, None, None]) -> str:
        output = ""
        for chunk in stream_response:
            output += chunk
        return output

    @staticmethod
    def get_args():
        file_path = sys.argv[0]
        line_number = int(sys.argv[1]) if len(sys.argv) > 1 else None
        selected_text = sys.argv[2] if len(sys.argv) > 2 else None

        if not selected_text:
            with open(file_path, 'r') as file:
                context = file.read()
        else:
            context = selected_text

        return {
            "context": context,
            "line_number": line_number,
        }

    def run(self):
        args_dict = self.get_args()
        context = args_dict["context"]
        logger.info("CONTEXT:")
        logger.debug(context)

        while True:
            seed = random.random()  # Generates a random float between 0 and 1

            time.sleep(1)
            logger.newline()

            prompt, model = self.get_user_input(context=context)

            logger.newline()
            logger.info("PROMPT:")
            logger.debug(prompt)

            logger.newline()
            logger.info("MODEL:")
            logger.debug(model)

            response = call_ollama_chat(
                prompt,
                model=model,
                options={
                    "seed": seed,
                },
            )
            output = self.handle_stream_response(response)
            # print(output)  # Output from the response


if __name__ == "__main__":
    script_path = os.path.abspath(__file__)

    # Start file observer for hot reload
    event_handler = HotReloadHandler(script_path)
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(
        script_path), recursive=False)
    observer.start()

    try:
        handler = ModelHandler()
        handler.run()
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
