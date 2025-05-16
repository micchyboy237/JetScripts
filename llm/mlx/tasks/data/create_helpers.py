import json
from pathlib import Path
import re
from typing import Dict, List, TypedDict, Optional
from uuid import uuid4
from jet.code.markdown_code_extractor import MarkdownCodeExtractor
from jet.llm.mlx.base import MLX
from jet.llm.mlx.mlx_types import Message, ModelKey
from jet.llm.mlx.utils import get_model_max_tokens
from jet.logger import logger
from jet.file.utils import load_file, save_file
import time

MODEL: ModelKey = "qwen3-8b-3bit"
seed = 42
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
    messages: List[Message]
    code: str
    error: Optional[str]
    duration: Optional[str]
    test_code: Optional[str]  # Added for test code
    test_duration: Optional[str]  # Added for test generation duration


def load_prompt_samples(file_path: str) -> List[PromptSample]:
    """Loads prompt samples from a markdown file containing JSON."""
    try:
        content = load_file(file_path)
        start = content.find('```json\n') + 8
        end = content.rfind('```')
        if start < 8 or end == -1:
            raise ValueError(
                "Invalid markdown format: JSON code block not found")

        json_str = content[start:end].strip()
        json_str = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
        json_str = re.sub(r'\s+', ' ', json_str)
        json_str = json_str.replace('\n', '')
        json_str = json_str.replace('\\n', '\\\\n')

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


def create_base_prompt() -> str:
    """Base prompt for Python script generation."""
    return (
        "As an expert Python developer, generate a complete, executable Python script adhering to PEP 8. "
        "Include docstrings and comments for clarity, avoid non-Python content, and end with 'TERMINATE' on a new line. "
        "If input is missing or unclear, add comments explaining assumptions."
    )


def create_code_prompt() -> str:
    """Prompt for generating Python code."""
    base = create_base_prompt()
    specific = (
        "Create a Python script based on the provided example or description. "
        "Use necessary imports, type hints, and try-except blocks for error handling. "
        "Ensure compatibility with the MLX framework (e.g., jet.llm.mlx) if specified. "
        "Match the input structure or functionality; if none provided, generate a minimal script."
    )
    return f"{base}\n{specific}"


def create_test_prompt() -> str:
    """Prompt for generating unit tests."""
    base = create_base_prompt()
    specific = (
        "Generate a unittest-based test script for the provided Python code. "
        "Use unittest.TestCase, test main functionality, edge cases, and errors. "
        "Include descriptive test names and docstrings. "
        "Mock dependencies (e.g., jet.llm.mlx) if present, using unittest.mock. "
        "Aim for comprehensive coverage. If no code is provided, assume a minimal function."
    )
    return f"{base}\n{specific}"


def generate_code_for_sample(sample: PromptSample, few_shot_examples: list[Message] = []) -> GeneratedCodeResult:
    """Generates Python code for a single prompt sample using the MLX model."""
    try:
        system_prompt = create_code_prompt()
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
            temperature=0.5,
            repetition_penalty=1.0,
            stop=["TERMINATE"],
        ):
            content = chunk["choices"][0]["message"]["content"]
            response += content
            logger.debug(content, flush=True)

        try:
            extractor = MarkdownCodeExtractor()
            code_block = extractor.extract_code_blocks(response)[0]
            code = code_block["code"]
        except:
            code = response

        return {
            "structure": sample["structure"],
            "system": system_prompt,
            "query": query,
            "code": code,
            "messages": messages,
            "error": None,
            "duration": None,
            "test_code": None,
            "test_duration": None
        }
    except Exception as e:
        logger.error(f"Error generating code for {sample['structure']}: {e}")
        return {
            "structure": sample["structure"],
            "system": system_prompt,
            "query": query,
            "messages": messages,
            "code": "",
            "error": str(e),
            "duration": None,
            "test_code": None,
            "test_duration": None
        }


def generate_tests_for_code(code: str, sample: PromptSample, few_shot_examples: list[Message] = []) -> tuple[str, Optional[str]]:
    """Generates unit tests for the provided Python code using the MLX model."""
    try:
        system_prompt = create_test_prompt()
        query = (
            f"Generate a unit test script for the following Python code:\n\n"
            f"{code}\n\n"
            f"The code implements the {sample['structure']} structure (Category: {sample['category']})."
        )
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
            temperature=0.5,
            repetition_penalty=1.0,
            stop=["TERMINATE"],
        ):
            content = chunk["choices"][0]["message"]["content"]
            response += content
            logger.debug(content, flush=True)
        if response.startswith('```python\n'):
            response = response[10:]
        if response.endswith('```'):
            response = response[:-3]
        return response.strip(), None
    except Exception as e:
        logger.error(f"Error generating tests for {sample['structure']}: {e}")
        return "", str(e)


def format_duration(seconds: float) -> str:
    """Converts duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes > 1 else ''} {remaining_seconds:.2f} seconds"
    hours = int(minutes // 60)
    remaining_minutes = minutes % 60
    return f"{hours} hour{'s' if hours > 1 else ''} {remaining_minutes} minute{'s' if remaining_minutes > 1 else ''}"


def create_helpers(output_dir: str, samples: List[PromptSample]) -> None:
    """Generates Python helper scripts and their unit tests for all prompt samples."""
    if not samples:
        logger.error("No prompt samples loaded. Exiting.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "code").mkdir(exist_ok=True)
    (output_path / "info").mkdir(exist_ok=True)
    (output_path / "tests").mkdir(exist_ok=True)  # Create tests directory

    sample_yes_no_answer_code = load_file(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/llm/mlx/tasks/yes_no_answer.py")
    sample_answer_multiple_choice_code = load_file(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/llm/mlx/tasks/answer_multiple_choice.py")
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

    # Mock test code for few-shot examples
    sample_yes_no_test_code = load_file(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/llm/mlx/tasks/test_yes_no_answer.py")

    sample_multiple_choice_test_code = load_file(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/llm/mlx/tasks/test_answer_multiple_choice.py")

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
            "content": sample_yes_no_answer_code
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
            "content": sample_answer_multiple_choice_code
        },
        # {
        #     "role": "user",
        #     "content": (
        #         f"Generate a unit test script for the following Python code:\n\n"
        #         f"{sample_answer_multiple_choice_code}\n\n"
        #         f"The code implements the {sample_answer_multiple_choice['structure']} structure (Category: {sample_answer_multiple_choice['category']})."
        #     ),
        # },
        # {
        #     "role": "assistant",
        #     "content": sample_multiple_choice_test_code
        # },
    ]

    for sample in samples:
        logger.info(
            f"Generating code for {sample['structure']} structure (Category: {sample['category']})")

        start_time = time.time()
        # Exclude test examples for code gen
        result = generate_code_for_sample(
            sample, few_shot_examples[:2] + few_shot_examples[3:4])
        end_time = time.time()
        duration_seconds = end_time - start_time
        result["duration"] = format_duration(duration_seconds)

        if result["error"]:
            logger.error(
                f"Failed to generate code for {sample['structure']}: {result['error']}")
            continue

        safe_structure = sample['structure'].lower().replace(
            '/', '_').replace(' ', '_')
        safe_category = sample['category'].lower().replace(
            ' ', '_').replace('/', '_')
        file_name = f"{safe_category}_{safe_structure}.py"
        code_file_path = output_path / "code" / file_name
        save_file(result["code"], str(code_file_path))

        # logger.info(
        #     f"Generating tests for {sample['structure']} structure (Category: {sample['category']})")
        # few_shot_test_examples: list[Message] = [
        #     {
        #         "role": "user",
        #         "content": (
        #             f"Generate a unit test script for the following Python code:\n\n"
        #             f"{sample_yes_no_answer_code}\n\n"
        #             f"The code implements the {sample_yes_no_answer['structure']} structure (Category: {sample_yes_no_answer['category']})."
        #         ),
        #     },
        #     {
        #         "role": "assistant",
        #         "content": sample_yes_no_test_code
        #     },
        #     {
        #         "role": "user",
        #         "content": (
        #             f"Generate a unit test script for the following Python code:\n\n"
        #             f"{sample_answer_multiple_choice_code}\n\n"
        #             f"The code implements the {sample_answer_multiple_choice['structure']} structure (Category: {sample_answer_multiple_choice['category']})."
        #         ),
        #     },
        #     {
        #         "role": "assistant",
        #         "content": sample_multiple_choice_test_code
        #     },
        # ]
        # start_time = time.time()
        # test_code, test_error = generate_tests_for_code(
        #     result["code"], sample, few_shot_test_examples)
        # end_time = time.time()
        # test_duration_seconds = end_time - start_time
        # result["test_code"] = test_code
        # result["test_duration"] = format_duration(test_duration_seconds)
        # if test_error:
        #     logger.error(
        #         f"Failed to generate tests for {sample['structure']}: {test_error}")
        #     result["error"] = test_error if not result[
        #         "error"] else f"{result['error']}; Test error: {test_error}"
        # if test_code:
        #     test_file_name = f"test_{safe_category}_{safe_structure}.py"
        #     test_file_path = output_path / "tests" / test_file_name
        #     save_file(test_code, str(test_file_path))

        messages = result.pop("messages")
        file_name = f"{safe_category}_{safe_structure}.json"
        info_file_path = output_path / "info" / file_name
        save_file(json.dumps(result, indent=2), str(info_file_path))

        file_name = f"{safe_category}_{safe_structure}.json"
        info_file_path = output_path / "messages" / file_name
        save_file(json.dumps(messages, indent=2), str(info_file_path))


if __name__ == "__main__":
    import os
    import shutil

    prompt_samples_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/llm/mlx/tasks/Dataset_Prompt_Samples.md"
    output_dir = os.path.join(
        os.path.dirname(__file__),
        "generated",
        os.path.splitext(os.path.basename(__file__))[0]
    )
    exclude_structures = ["Yes/No", "Multiple Choice"]

    os.makedirs(output_dir, exist_ok=True)

    samples = load_prompt_samples(prompt_samples_file)
    samples = [s for s in samples if s["structure"] not in exclude_structures]
    create_helpers(output_dir, samples)
