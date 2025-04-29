import argparse
import json
import os
from jet.file.utils import save_file
import lm_eval
import numpy as np
from pathlib import Path
from lm_eval.api.instance import Instance
from tqdm import tqdm
from jet.logger import CustomLogger

# Assuming MLXLM and other dependencies are available from the provided code
from mlx_lm.evaluate import MLXLM  # Replace with actual import path

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")


def evaluate_on_benchmarks(model_path, tasks, batch_size=32, num_shots=5, output_dir="./eval_results"):
    """
    Evaluate a language model on standard NLP benchmarks like MMLU or HellaSwag.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = MLXLM(model_path, batch_size=batch_size, use_chat_template=True)
    results = lm_eval.simple_evaluate(
        model=model,
        tasks=tasks,
        num_fewshot=num_shots,
        apply_chat_template=model.use_chat_template,
        random_seed=123,
        numpy_random_seed=123,
        torch_random_seed=123,
        fewshot_random_seed=123,
    )

    model_name = model_path.replace("/", "_")
    task_names = "_".join(tasks)
    filename = f"eval_{model_name}_{task_names}_{num_shots:02d}_v_lm_eval.json"
    output_path = output_dir / filename
    save_file(results, output_path)
    return results["results"]


def compute_log_likelihood(model_path, contexts, continuations, batch_size=16):
    """
    Compute log-likelihood for text completion tasks, e.g., ranking multiple-choice answers.
    """
    model = MLXLM(model_path, batch_size=batch_size)
    requests = [
        Instance(
            request_type="loglikelihood",
            doc={"context": ctx, "continuation": cont},
            idx=i,
            arguments=(ctx, cont)
        )
        for i, (ctx, cont) in enumerate(zip(contexts, continuations))
    ]
    results = model.loglikelihood(requests)

    output = []
    for (logprob, is_greedy), req in zip(results, requests):
        output.append({
            "context": req.args[0],
            "continuation": req.args[1],
            "logprob": float(logprob),
            "is_greedy": bool(is_greedy)
        })
    return output


def generate_until_stop(model_path, prompts, until_conditions, max_gen_tokens=100, batch_size=16):
    """
    Generate text until a stopping condition is met, e.g., for structured content creation.
    """
    model = MLXLM(model_path, batch_size=batch_size)
    requests = [
        Instance(
            request_type="generate",
            doc={"prompt": prompt, "until": until,
                 "max_gen_tokens": max_gen_tokens},
            idx=i,
            arguments=(prompt, {"until": until,
                       "max_gen_tokens": max_gen_tokens})
        )
        for i, (prompt, until) in enumerate(zip(prompts, until_conditions))
    ]
    completions = model.generate_until(requests)
    return completions


def measure_perplexity(model_path, texts, batch_size=16):
    """
    Measure perplexity to assess text quality, e.g., for product descriptions.
    """
    model = MLXLM(model_path, batch_size=batch_size)
    requests = [
        Instance(
            request_type="loglikelihood_rolling",
            doc={"text": text},
            idx=i,
            arguments=(text,)
        )
        for i, text in enumerate(texts)
    ]
    scores = model.loglikelihood_rolling(requests)

    output = []
    for score, req in zip(scores, requests):
        token_count = len(model._tokenize([req.args[0]])[0])
        perplexity = np.exp(-score /
                            token_count) if token_count > 0 else float("inf")
        output.append({
            "text": req.args[0],
            "perplexity": float(perplexity)
        })
    return output


def few_shot_evaluation(model_path, task, num_shots=3, batch_size=16, output_dir="./eval_results"):
    """
    Evaluate a model on a custom task with few-shot learning.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = MLXLM(model_path, batch_size=batch_size, use_chat_template=True)
    results = lm_eval.simple_evaluate(
        model=model,
        tasks=[task],
        num_fewshot=num_shots,
        fewshot_as_multiturn=True,
        apply_chat_template=model.use_chat_template,
        random_seed=123,
        numpy_random_seed=123,
        torch_random_seed=123,
        fewshot_random_seed=123,
    )

    model_name = model_path.replace("/", "_")
    filename = f"eval_{model_name}_{task}_{num_shots:02d}_v_lm_eval.json"
    output_path = output_dir / filename
    save_file(results, output_path)
    return results["results"]


def distributed_evaluation(model_path, task, batch_size=64, limit=10000, output_dir="./eval_results"):
    """
    Perform distributed evaluation across multiple GPUs for large datasets.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = MLXLM(model_path, batch_size=batch_size)
    results = lm_eval.simple_evaluate(
        model=model,
        tasks=[task],
        limit=limit,
        apply_chat_template=model.use_chat_template,
        random_seed=123,
        numpy_random_seed=123,
        torch_random_seed=123,
        fewshot_random_seed=123,
    )

    model_name = model_path.replace("/", "_")
    filename = f"eval_{model_name}_{task}_00_v_lm_eval.json"
    output_path = output_dir / filename
    save_file(results, output_path)
    return results["results"]


def main():
    # Example model path (replace with actual model path or HF repo)
    model_path = "mlx-community/Llama-3.2-3B-Instruct-4bit"

    # # 1. Evaluate on benchmarks
    # logger.info("Running benchmark evaluation...")
    # benchmark_results = evaluate_on_benchmarks(
    #     model_path=model_path,
    #     tasks=["mmlu", "hellaswag"],
    #     batch_size=32,
    #     num_shots=5
    # )
    # logger.log("\nBenchmark Results:", json.dumps(
    #     benchmark_results, indent=2), colors=["GRAY", "SUCCESS"])

    # 2. Compute log-likelihood
    logger.info("\nComputing log-likelihood...")
    log_likelihood_results = compute_log_likelihood(
        model_path=model_path,
        contexts=["What is the capital of France? "] * 2,
        continuations=["Paris", "Florida"],
        batch_size=16
    )
    logger.log("\nLog-Likelihood Results:",
               json.dumps(log_likelihood_results, indent=2), colors=["GRAY", "SUCCESS"])

    # 3. Generate until stop
    logger.info("\nGenerating text until stop...")
    generated_texts = generate_until_stop(
        model_path=model_path,
        prompts=["Write a short story about a cat: "],
        until_conditions=[["\n\n"]],
        max_gen_tokens=100,
        batch_size=16
    )
    logger.log("\nGenerated Texts:", generated_texts,
               colors=["GRAY", "SUCCESS"])

    # 4. Measure perplexity
    logger.info("\nMeasuring perplexity...")
    perplexity_results = measure_perplexity(
        model_path=model_path,
        texts=[
            "This is a well-written product description.",
            "This bad write product descr."
        ],
        batch_size=16
    )
    logger.log("\nPerplexity Results:", json.dumps(
        perplexity_results, indent=2), colors=["GRAY", "SUCCESS"])

    # # 5. Few-shot evaluation
    # logger.info("\nRunning few-shot evaluation...")
    # few_shot_results = few_shot_evaluation(
    #     model_path=model_path,
    #     task="custom_task",  # Replace with actual task name
    #     num_shots=3,
    #     batch_size=16
    # )
    # logger.log("\nFew-Shot Results:", json.dumps(few_shot_results,
    #                                              indent=2), colors=["GRAY", "SUCCESS"])

    # # 6. Distributed evaluation
    # logger.info("\nRunning distributed evaluation...")
    # distributed_results = distributed_evaluation(
    #     model_path=model_path,
    #     task="wikitext",
    #     batch_size=64,
    #     limit=10000
    # )
    # logger.log("\nDistributed Evaluation Results:", json.dumps(
    #     distributed_results, indent=2), colors=["GRAY", "SUCCESS"])


if __name__ == "__main__":
    main()
