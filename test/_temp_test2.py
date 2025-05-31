import argparse
import json
from pathlib import Path


def parse_evaluation_args(
    model: str = None,
    tasks: list = None,
    output_dir: str = ".",
    batch_size: int = 16,
    num_shots: int = None,
    max_tokens: int = None,
    limit: int = None,
    seed: int = 123,
    fewshot_as_multiturn: bool = False,
    apply_chat_template: bool = None,
    chat_template_args: str = "{}"
) -> argparse.Namespace:
    """
    Parse arguments for MLX model evaluation using lm-evaluation-harness.

    Args:
        model: Model to evaluate
        tasks: List of tasks to evaluate
        output_dir: Output directory for result files
        batch_size: Batch size for evaluation
        num_shots: Number of few-shot examples
        max_tokens: Maximum number of tokens to generate
        limit: Limit the number of examples per task
        seed: Random seed
        fewshot_as_multiturn: Whether to provide fewshot examples as multiturn conversation
        apply_chat_template: Whether to apply chat template to prompt
        chat_template_args: JSON string of arguments for tokenizer's apply_chat_template

    Returns:
        argparse.Namespace containing the parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Evaluate an MLX model using lm-evaluation-harness."
    )

    parser.add_argument("--model", help="Model to evaluate",
                        default=model, required=not model)
    parser.add_argument(
        "--tasks", nargs="+", help="Tasks to evaluate", default=tasks, required=not tasks)
    parser.add_argument(
        "--output-dir",
        default=output_dir,
        help="Output directory for result files."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=batch_size,
        help="Batch size"
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        default=num_shots,
        help="Number of shots"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=max_tokens,
        help="Maximum number of tokens to generate. Defaults to the model's max context length."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=limit,
        help="Limit the number四大号 of examples per task."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=seed,
        help="Random seed."
    )
    parser.add_argument(
        "--fewshot-as-multiturn",
        action="store_true",
        help="Whether to provide the fewshot examples as a multiturn conversation or a single user turn.",
        default=fewshot_as_multiturn
    )
    parser.add_argument(
        "--apply-chat-template",
        action=argparse.BooleanOptionalAction,
        help="Specifies whether to apply a chat template to the prompt.",
        default=apply_chat_template
    )
    parser.add_argument(
        "--chat-template-args",
        type=json.loads,
        help="A JSON formatted string of arguments for the tokenizer's apply_chat_template",
        default=chat_template_args
    )

    return parser.parse_args()


# Usage example
if __name__ == "__main__":
    # Example 1: Basic usage with required arguments
    args1 = parse_evaluation_args(
        model="mistral-7b",
        tasks=["hellaswag", "arc_challenge"],
        output_dir="./results"
    )
    print("Example 1 args:", vars(args1))

    # Example 2: Full configuration
    args2 = parse_evaluation_args(
        model="llama-3b",
        tasks=["mmlu", "gsm8k"],
        output_dir="./eval_results",
        batch_size=32,
        num_shots=5,
        max_tokens=2048,
        limit=100,
        seed=42,
        fewshot_as_multiturn=True,
        apply_chat_template=True,
        chat_template_args='{"enable_thinking": true}'
    )
    print("Example 2 args:", vars(args2))
