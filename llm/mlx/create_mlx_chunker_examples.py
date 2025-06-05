import os
from pathlib import Path
from typing import Generator, TypedDict

from jet.code.markdown_code_extractor import MarkdownCodeExtractor
from jet.file.utils import save_file
from jet.llm.mlx.base import MLX
from jet.llm.mlx.mlx_types import LLMModelKey
from jet.logger import logger

MODEL: LLMModelKey = "llama-3.2-3b-instruct-4bit"

SYSTEM_PROMPT = """
"""

PROMPT_TEMPLATE = """
Generate multiple usage examples given the code below. Use "from chunker" to import. Add an "__main__" condition at the end to run all examples. Surround all in a single python code block.

# Code
{chunker_code}
"""

mlx = MLX(MODEL)


class GenerationResult(TypedDict):
    context: str
    response: str


def call_llm_generation(context: str) -> GenerationResult:
    response = ""
    for chunk in mlx.stream_chat(
        context,
        max_tokens=-1,
        system_prompt=SYSTEM_PROMPT,
        temperature=0.3,
    ):
        content = chunk["choices"][0]["message"]["content"]
        response += content
        logger.success(content, flush=True)
    return {
        "context": context,
        "response": response,
    }


class GeneratorGenerationResult(TypedDict):
    source: str
    code: str


def run_generate_examples(base_dir: str) -> Generator[GeneratorGenerationResult, None, None]:
    # Find all chunker.py files under base_dir and sort them
    chunker_files = sorted(Path(base_dir).rglob("chunker.py"))
    logger.debug(f"Chunker files: {len(chunker_files)}")

    for chunker_file in chunker_files:
        logger.info(f"\nProcessing {chunker_file}")
        # Load the content of chunker.py
        try:
            with open(chunker_file, "r", encoding="utf-8") as f:
                chunker_code = f.read()
        except Exception as e:
            logger.error(f"Failed to read {chunker_file}: {e}")
            continue

        # Format the prompt with the chunker.py content
        context = PROMPT_TEMPLATE.format(chunker_code=chunker_code)
        extractor = MarkdownCodeExtractor()
        result = call_llm_generation(context)
        code_blocks = extractor.extract_code_blocks(result["response"])
        code = "\n\n".join(code_block["code"] for code_block in code_blocks)
        yield {
            "source": str(chunker_file),
            "code": code
        }


if __name__ == "__main__":
    base_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/mlx/chunker_classes"

    results = run_generate_examples(base_dir)

    for result in results:
        # Save examples.py in the same directory as chunker.py
        output_dir = os.path.dirname(result["source"])
        logger.newline()
        save_file(result["code"], os.path.join(
            output_dir, "chunker_examples.py"))
