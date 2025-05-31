# Usage example
import os
from jet.file.utils import save_file
from jet.llm.mlx.evaluation.evaluate_model import parse_evaluation_args, evaluate_model


if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

    # Example 1: Basic usage with required arguments
    args1 = parse_evaluation_args(
        model="mistral-7b",
        tasks=["hellaswag", "arc_challenge"],
        output_dir="./results"
    )
    print("Example 1 args:", vars(args1))
    eval_results1 = evaluate_model(args1)
    save_file(eval_results1, f"{output_dir}/eval_results1.json")

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
    eval_results2 = evaluate_model(args1)
    save_file(eval_results2, f"{output_dir}/eval_results2.json")
