import os
from llm.eval.convert_notebooks_to_scripts import scrape_notes

from jet.llm import call_ollama_chat
from jet.llm.llm_types import OllamaChatOptions
from jet.logger import logger

# Define input directory
input_dir = "/Users/jethroestrada/Desktop/External_Projects/JetScripts/llm/eval/notebooks"
generated_dir = "descriptions"

MODEL = "llama3.1"
SYSTEM_MESSAGE = "You are an AI assistant that specializes in documenting code in a readable and organized format using markdown. You can explain code written in any language, summarize its features, provide insights into its real-world applications with code usage samples."

CHAT_OPTIONS: OllamaChatOptions = {
    "seed": 43,
    "num_ctx": 4096,
    "num_keep": 0,
    "num_predict": -1,
    "temperature": 0,
}
INSTRUCTIONS = """
Write an organized documentation of this code using markdown.
""".strip()
PROMPT_TEMPLATE = "{instructions}\n\nCode:\n\n```python\n{code}\n```\n\nResponse:\n"
FINAL_MARKDOWN_TEMPLATE = "## System\n\n```\n{system}\n```\n\n## Prompt\n\n```\n{prompt}\n```\n\n## Response\n\n{response}"

# Read .ipynb files
files = [os.path.join(input_dir, f)
         for f in os.listdir(input_dir) if f.endswith(".py")]
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
            "experiment": "Code Description Test",
            "run_name": "Describe python code",
            "format": FINAL_MARKDOWN_TEMPLATE,
            "metadata": {
                "type": "code_description",
            }
        }
    ):
        response += chunk
        logger.success(chunk, flush=True)
    return response


if __name__ == "__main__":
    scrape_notes(with_markdown=True)
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
            output_file = os.path.join(output_dir, f"{file_name}.md")

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)
            logger.log("Saved to", output_file, colors=[
                "WHITE", "BRIGHT_SUCCESS"])

        except Exception as e:
            print(f"Failed to process file {file_name}: {e}")

    print(f"Total files processed: {len(files)}")
