import json
from pathlib import Path
import re
from typing import Dict, List, TypedDict, Optional
from uuid import uuid4
from jet.llm.mlx.base import MLX
from jet.llm.mlx.mlx_types import Message, ModelKey
from jet.llm.mlx.utils import get_model_max_tokens
from jet.logger import logger
from jet.file.utils import load_file, save_file

seed = 42
MODEL: ModelKey = "qwen3-8b-3bit"
mlx = MLX(MODEL, seed=seed)


class PromptSample(TypedDict):
    category: str
    structure: str
    system_message: str
    input: str
    output: str


class GeneratedCodeResult(TypedDict):
    structure: str
    system: str
    query: str
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
        json_str = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
        json_str = re.sub(r'\s+', ' ', json_str)
        json_str = json_str.replace('\n', '')
        json_str = json_str.replace('\\n', '\\\\n')

        # Parsecandidates JSON
        samples = json.loads(json_str)
        if not isinstance(samples, list):
            raise ValueError("Parsed JSON is not a list of prompt samples")

        logger.success(
            f"Loaded {len(samples)} prompt samples from {file_path}")
        return samples
    except Exception as e:
        logger.error(
            f"Error loading prompt samples from {file_path}: {str(e)}")
        logger.debug(f"Problematic JSON string: {json_str[:100]}...")
        return []


def create_system_prompt_for_code_generation() -> str:
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
        "Generate a complete Python script that implements the provided info. Do not include markdown code fences or any non-Python content. "
        "Ensure the script can be saved and run directly."
    )
    return base_prompt


def generate_code_for_sample(sample: PromptSample, few_shot_examples: list[Message] = []) -> GeneratedCodeResult:
    """Generates Python code for a single prompt sample using the MLX model."""
    try:
        system_prompt = create_system_prompt_for_code_generation()
        query = f"Generate a Python script for the {sample['structure']} structure."
        response = ""

        messages: list[Message] = [
            {"role": "system", "content": system_prompt},
            *few_shot_examples,
            {"role": "user", "content": query},
        ]
        max_tokens = get_model_max_tokens(MODEL)
        for chunk in mlx.stream_chat(
            messages,
            max_tokens=max_tokens * 2,
            temperature=0.3,
            stop=["\n```"]
        ):
            content = chunk["choices"][0]["message"]["content"]
            response += content
            logger.debug(content, flush=True)
        # Clean response to remove any markdown or non-code content
        if response.startswith('```python\n'):
            response = response[10:]
        if response.endswith('```'):
            response = response[:-3]
        return {
            "structure": sample["structure"],
            "system": system_prompt,
            "query": query,
            "code": response.strip(),
            "error": None
        }
    except Exception as e:
        logger.error(f"Error generating code for {sample['structure']}: {e}")
        return {
            "structure": sample["structure"],
            "system": system_prompt,
            "query": query,
            "code": "",
            "error": str(e)
        }


def create_helpers(output_dir: str, prompt_samples_file: str = "Dataset_Prompt_Samples.md") -> None:
    """Generates Python helper scripts for all prompt samples."""
    samples = load_prompt_samples(prompt_samples_file)
    if not samples:
        logger.error("No prompt samples loaded. Exiting.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sample_yes_no_answer_code = load_file(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/llm/mlx/helpers/yes_no_answer.py")
    sample_answer_multiple_choice_code = load_file(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/llm/mlx/helpers/answer_multiple_choice.py")
    sample_yes_no_answer = {
        "category": "Classification",
        "structure": "Yes/No",
        "system_message": "Answer the question with 'Yes' or 'No' only.",
        "input": "Is Python a programming language?",
        "output": "Yes",
    }
    sample_answer_multiple_choice = {
        "category": "Selection",
        "structure": "Multiple Choice",
        "system_message": "Answer the following question by choosing one of the options provided without any additional text.\nOptions:\nMars\nEarth\nJupiter\nSaturn",
        "input": "Which planet is known as the Red Planet?",
        "output": "Mars"
    }
    few_shot_examples: list[Message] = [
        {
            "role": "user",
            "content": (
                f"Structure: {sample_yes_no_answer['structure']}\n"
                f"Category: {sample_yes_no_answer['category']}\n"
                f"System Message: {sample_yes_no_answer['system_message']}\n"
                f"Input: {sample_yes_no_answer['input']}\n"
                f"Output: {sample_yes_no_answer['output']}\n"
                "Response:"
            ),
        },
        {
            "role": "assistant",
            "content": f"```python\n{sample_yes_no_answer_code}\n```"
        },
        {
            "role": "user",
            "content": (
                f"Structure: {sample_answer_multiple_choice['structure']}\n"
                f"Category: {sample_answer_multiple_choice['category']}\n"
                f"System Message: {sample_answer_multiple_choice['system_message']}\n"
                f"Input: {sample_answer_multiple_choice['input']}\n"
                f"Output: {sample_answer_multiple_choice['output']}\n"
                "Response:"
            ),
        },
        {
            "role": "assistant",
            "content": f"```python\n{sample_answer_multiple_choice_code}\n```"
        },
    ]

    for sample in samples:
        logger.info(
            f"Generating code for {sample['structure']} structure (Category: {sample['category']})")
        result = generate_code_for_sample(sample, few_shot_examples)

        if result["error"]:
            logger.error(
                f"Failed to generate code for {sample['structure']}: {result['error']}")
            continue

        # Generate a safe filename based on structure and category
        safe_structure = sample['structure'].lower().replace(
            '/', '_').replace(' ', '_')
        safe_category = sample['category'].lower().replace(
            ' ', '_').replace('/', '_')
        file_name = f"{safe_category}_{safe_structure}.py"

        code_file_path = output_path / "code" / file_name
        save_file(result["code"], str(code_file_path))

        file_name = f"{safe_category}_{safe_structure}.json"
        info_file_path = output_path / "info" / file_name
        save_file(result, str(info_file_path))


if __name__ == "__main__":
    import os
    import shutil

    prompt_samples_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/llm/mlx/helpers/Dataset_Prompt_Samples.md"
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(
            os.path.basename(__file__))[0]
    )

    os.makedirs(output_dir, exist_ok=True)

    create_helpers(output_dir, prompt_samples_file)
