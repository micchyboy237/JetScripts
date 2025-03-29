import os
from llm.eval.convert_notebooks_to_scripts import scrape_notes

from jet.actions.generation import call_ollama_chat
from jet.llm.llm_types import OllamaChatOptions
from jet.logger import logger

# Define input directory
input_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/eval/notebooks"
exclude_files = [
    "answer_and_context_relevancy",
    "semantic_similarity_eval",
]
generated_dir = "improved"

MODEL = "codellama"
SYSTEM_MESSAGE = "You are an AI assistant that follows instructions. You can understand and write code of any language, extract code from structured and unstructured content, and provide real-world usage examples. You can write clean, optimized, readable, and modular code. You follow best practices and correct syntax."
CHAT_OPTIONS: OllamaChatOptions = {
    "seed": 43,
    "num_ctx": 4096,
    "num_keep": 0,
    "num_predict": -1,
    "temperature": 0,
}
INSTRUCTIONS = """
Refactor this code as classes with types improved for readability, modularity, and reusability.
Add main function for real world usage examples.
Generated code should be complete and working with correct syntax.

Respond only with a single Python code wrapped in a code block without additional information (use ```python).
""".strip()
PROMPT_TEMPLATE = "{instructions}\n\nCode:\n\n```python\n{code}\n```\n\nResponse:\n"
FINAL_MARKDOWN_TEMPLATE = "## System\n\n```\n{system}\n```\n\n## Prompt\n\n```\n{prompt}\n```\n\n## Response\n\n{response}"

# Read .py files
files = [os.path.join(input_dir, f)
         for f in os.listdir(input_dir) if f.endswith(".py")]
filtered_files = [
    file for file in files
    if not any(exclude in file for exclude in exclude_files)
]
files = filtered_files
print(f"Found {len(files)} .ipynb files: {files}")

# Function to extract Python code cells from a file


def read_file(file):
    with open(file, 'r', encoding='utf-8') as f:
        source = f.read()
    return source


def improve_code(code):
    logger.debug("Generating response...")
    prompt = PROMPT_TEMPLATE.format(
        instructions=INSTRUCTIONS,
        code=code,
    )
    logger.log("PROMPT:")
    logger.info(prompt)

    response = ""
    for chunk in call_ollama_chat(
        prompt,
        stream=True,
        model=MODEL,
        system=SYSTEM_MESSAGE,
        options=CHAT_OPTIONS,
        track={
            "repo": "./aim-logs",
            "experiment": "Code Enhancer Test",
            "run_name": "Improve python",
            "format": FINAL_MARKDOWN_TEMPLATE,
            "metadata": {
                "type": "code_enhancer",
            }
        }
    ):
        response += chunk
        logger.success(chunk, flush=True)
    return response


if __name__ == "__main__":
    scrape_notes(with_markdown=False)

    # Process each file and extract Python code
    for file in files:
        file_path = os.path.join(input_dir, file)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"Processing file: {file_name}...")

        try:
            content = read_file(file_path)

            response = improve_code(content)

            output_dir = os.path.join(os.path.dirname(__file__), generated_dir)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{file_name}.py")

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)

        except Exception as e:
            print(f"Failed to process file {file_name}: {e}")

    print(f"Total files processed: {len(files)}")
