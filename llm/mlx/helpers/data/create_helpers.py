import json
from pathlib import Path
import re
from typing import Dict, List, TypedDict, Optional
from uuid import uuid4
from jet.llm.mlx.base import MLX
from jet.llm.mlx.mlx_types import ModelKey
from jet.logger import logger
from jet.file.utils import load_file, save_file

MODEL: ModelKey = "llama-3.2-3b-instruct-4bit"
mlx = MLX(MODEL)


class PromptSample(TypedDict):
    category: str
    structure: str
    system_message: str
    input: str
    output: str


class GeneratedCodeResult(TypedDict):
    structure: str
    code: str
    error: Optional[str]


def load_prompt_samples(file_path: str) -> List[PromptSample]:
    """Loads prompt samples from a markdown file containing JSON."""
    try:
        content = load_file(file_path)
        # Extract JSON from markdown code block
        start = content.find('```json\n') + 8
        end = content.rfind('```')
        if start < 8 or end == -1:
            raise ValueError(
                "Invalid markdown format: JSON code block not found")

        json_str = content[start:end].strip()
        # Clean JSON string: remove invalid control characters and normalize whitespace
        # Remove control characters
        json_str = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
        json_str = re.sub(r'\s+', ' ', json_str)  # Normalize whitespace
        json_str = json_str.replace('\n', '')  # Remove newlines within JSON
        json_str = json_str.replace('\\n', '\\\\n')  # Escape newlines properly

        # Parse JSON
        samples = json.loads(json_str)
        if not isinstance(samples, list):
            raise ValueError("Parsed JSON is not a list of prompt samples")

        logger.success(
            f"Loaded {len(samples)} prompt samples from {file_path}")
        return samples
    except Exception as e:
        logger.error(
            f"Error loading prompt samples from {file_path}: {str(e)}")
        # Log first 100 chars for debugging
        logger.debug(f"Problematic JSON string: {json_str[:100]}...")
        return []


def create_system_prompt_for_code_generation(sample: PromptSample) -> str:
    """Creates an optimized system prompt for generating Python code based on the sample."""
    base_prompt = (
        "You are an expert Python developer tasked with generating complete, functional, and well-documented Python code. "
        "Based on the provided example, create a Python script that implements the specified structure. The code should:\n"
        "- Be syntactically correct and follow PEP 8 style guidelines.\n"
        "- Include necessary imports and type hints where applicable.\n"
        "- Handle errors gracefully with try-except blocks.\n"
        "- Include docstrings and comments for clarity.\n"
        "- Be compatible with the existing MLX framework (e.g., use jet.llm.mlx modules).\n"
        "- Produce output matching the example structure exactly.\n\n"
        "Example provided:\n"
        f"Structure: {sample['structure']}\n"
        f"System Message: {sample['system_message']}\n"
        f"Input: {sample['input']}\n"
        f"Output: {sample['output']}\n\n"
        "Generate a complete Python script that implements this structure. Do not include markdown code fences or any non-Python content. "
        "Ensure the script can be saved and run directly."
    )
    return base_prompt


def generate_code_for_sample(sample: PromptSample) -> GeneratedCodeResult:
    """Generates Python code for a single prompt sample using the MLX model."""
    try:
        system_prompt = create_system_prompt_for_code_generation(sample)
        context = f"Generate a Python script for the {sample['structure']} structure."
        response = ""
        for chunk in mlx.stream_chat(
            context,
            max_tokens=-1,
            system_prompt=system_prompt,
            temperature=0.3,
        ):
            content = chunk["choices"][0]["message"]["content"]
            response += content
            logger.success(content, flush=True)
        # Clean response to remove any markdown or non-code content
        if response.startswith('```python\n'):
            response = response[10:]
        if response.endswith('```'):
            response = response[:-3]
        return {
            "structure": sample["structure"],
            "code": response.strip(),
            "error": None
        }
    except Exception as e:
        logger.error(f"Error generating code for {sample['structure']}: {e}")
        return {
            "structure": sample["structure"],
            "code": "",
            "error": str(e)
        }


def create_helpers(output_dir: str, prompt_samples_file: str = "Dataset_Prompt_Samples.md") -> None:
    """Generates Python helper scripts for Yes/No and Multiple Choice structures from prompt samples."""
    samples = load_prompt_samples(prompt_samples_file)
    if not samples:
        logger.error("No prompt samples loaded. Exiting.")
        return

    # Filter for Yes/No and Multiple Choice structures
    target_structures = ["Yes/No", "Multiple Choice"]
    relevant_samples = [
        s for s in samples if s["structure"] in target_structures]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for sample in relevant_samples:
        logger.info(f"Generating code for {sample['structure']} structure")
        result = generate_code_for_sample(sample)

        if result["error"]:
            logger.error(
                f"Failed to generate code for {sample['structure']}: {result['error']}")
            continue

        # Generate a safe filename based on structure
        file_name = f"{sample['structure'].lower().replace('/', '_').replace(' ', '_')}.py"
        file_path = output_path / file_name

        try:
            save_file(result["code"], str(file_path))
            logger.success(f"Saved generated code to {file_path}")
        except Exception as e:
            logger.error(f"Error saving file {file_path}: {e}")


if __name__ == "__main__":
    import os
    prompt_samples_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/llm/mlx/helpers/Dataset_Prompt_Samples.md"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", "helpers"
    )

    create_helpers(output_dir, prompt_samples_file)
