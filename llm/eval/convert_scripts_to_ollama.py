import os
from llm.eval.convert_notebooks_to_scripts import scrape_notes
from jet.llm import call_ollama_chat
from jet.llm.llm_types import OllamaChatOptions
from jet.logger import logger


# Define input directory
input_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/eval/notebooks"
exclude_files = [
    "answer_and_context_relevancy",
    "semantic_similarity_eval",
]
generated_dir = "ollama_notebooks"

MODEL = "codellama"
SYSTEM_MESSAGE = ""
CHAT_OPTIONS: OllamaChatOptions = {
    "seed": 44,
    "num_ctx": 4096,
    "num_keep": 0,
    "num_predict": -1,
    "temperature": 0,
}
INSTRUCTIONS = """
You are given a code that will be updated based on the ff:

1. Refactor with main. Initialize ollama.
- Add a main function to contain all usage examples.
- Add this import line at the top `from jet.llm.ollama import initialize_ollama_settings`
- Call this at the top of main function `initialize_ollama_settings()` before any code.

2. Replace openai llm and embed models with ollama. Use something like fnmatch if needed. Here are some guidelines:
- Replace `from llama_index.llms.openai import OpenAI` to `from llama_index.llms.ollama import Ollama`
- Replace any model that starts with gpt-4 becomes `model="llama3.1"`. Ex: `model="gpt-4-turbo"`
- Replace any model that starts with gpt-3.5 becomes `model="llama3.1"`. Ex: `model="gpt-3.5-turbo"`
- Replace `from llama_index.embeddings.openai import OpenAIEmbedding` to `from llama_index.embeddings.ollama import OllamaEmbedding`
- Replace any call to OpenAIEmbeddings(**) with OllamaEmbedding(model_name="mxbai-embed-large", base_url="http://localhost:11434")
- Update variable names appropriately since ollama doesn't have gpt models.

3. Follow this code guidelines:
- Keep the code short, reusable, testable, maintainable and optimized. Follow best practices and industry design patterns.
- Organize the code with proper structure and spacing.
- Separate usage examples as functions that will be called in main.

4. Respond only with a single Python code wrapped in a code block without additional information (use ```python).
""".strip()
PROMPT_TEMPLATE = "Instructions:\n\n{instructions}\n\nCode:\n\n```python\n{code}\n```\n\nResponse:\n"
FINAL_MARKDOWN_TEMPLATE = "## System\n\n```\n{system}\n```\n\n## Prompt\n\n```\n{prompt}\n```\n\n## Response\n\n{response}"

# Read .py files
files = [os.path.join(input_dir, f)
         for f in os.listdir(input_dir) if f.endswith(".py")]
filtered_files = [
    file for file in files
    if not any(exclude in file for exclude in exclude_files)
]
files = filtered_files
print(f"Found {len(files)} .py files")

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
    # scrape_notes(with_markdown=False)

    # Process each file and extract Python code
    for file in files:
        file_path = os.path.join(input_dir, file)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        logger.log("Processing file", file_path,
                   colors=["GRAY", "BRIGHT_DEBUG"])

        try:
            content = read_file(file_path)

            response = improve_code(content)

            output_dir = os.path.join(os.path.dirname(__file__), generated_dir)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{file_name}.py")

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(response)

            logger.log("Saved to", output_file, colors=[
                       "WHITE", "BRIGHT_SUCCESS"])

        except Exception as e:
            logger.error(f"Failed to process file {file_name}: {e}")

    logger.log("Total files processed:", len(
        files), colors=["WHITE", "SUCCESS"])
