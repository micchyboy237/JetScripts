import os
from jet.llm import call_ollama_chat
from jet.llm.llm_types import OllamaChatOptions
from jet.logger import logger
from jet_template_combined import generate_improve_prompt

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

config_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/llm/ollama/config.py"
with open(config_path, 'r') as file:
    config_contents = file.read()

config_context = f"""
Use this config for llm and embed models to replace openai usage.

```python
{config_contents}
```
""".strip()

RAG_TEMPLATE = """
Use the following context as your learned knowledge, inside <context></context> XML tags.
<context>
    {context}
</context>

Replace openai imports and usage with ollama based from context.
You may replace openai with ollama like this:

Sample OpenAI code:
llm = OpenAI(temperature=0, model="gpt-4")
embed_model = OpenAIEmbedding(embed_batch_size=batch_size)

Replace with ollama llm and embedding:
from jet.llm.ollama import (
    update_llm_settings,
    large_llm_model,
    large_embed_model,
)
settings = update_llm_settings(
    "llm_model": large_llm_model,
    "embedding_model": large_embed_model,
)
llm = settings.llm
embed_model = settings.embed_model
""".strip()

INSTRUCTIONS = """
Analyze the context and follow its rules.
Refactor this code as classes with types improved for readability, modularity, and reusability.
Add main function for real world usage examples.
Generated code should be complete and working with correct syntax.

Respond only with a single Python code wrapped in a code block without additional information (use ```python).
""".strip()
PROMPT_TEMPLATE = "Context:\n\n{context}\n\nInstructions:\n\n{instructions}\n\nCode:\n\n```python\n{code}\n\n```\n\nResponse:\n"
FINAL_MARKDOWN_TEMPLATE = "## System\n\n```\n{system}\n```\n\n## Prompt\n\n```\n{prompt}\n\n```\n\n## Response\n\n{response}"

# Read .ipynb files
files = [os.path.join(input_dir, f)
         for f in os.listdir(input_dir) if f.endswith(".py")]
filtered_files = [
    file for file in files
    if not any(exclude in file for exclude in exclude_files)
]
files = filtered_files
print(f"Found {len(files)} .ipynb files")

# Function to extract Python code cells from a file


def read_file(file):
    with open(file, 'r', encoding='utf-8') as f:
        source = f.read()
    return source


def improve_code(code):
    logger.debug("Generating response...")
    prompt = PROMPT_TEMPLATE.format(
        context=RAG_TEMPLATE.format(context=config_context),
        instructions=INSTRUCTIONS,
        code=code,
    )
    # prompt = generate_improve_prompt(code, prompt=INSTRUCTIONS)
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
        print("Saved to " + output_file)

    except Exception as e:
        print(f"Failed to process file {file_name}: {e}")

print(f"Total files processed: {len(files)}")
