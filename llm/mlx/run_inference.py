"""
Run parallel inference with the MLX framework for the Llama-3.2-1B-Instruct-4bit model using mpi4py.

This script supports generating responses from a list of prompts (via --promptfile or --prompt),
limiting parallel model instances (--num-parallel), and saving responses to a directory (--output-dir).
It uses mpi4py for parallel processing, launching processes via mpirun or mpiexec.

Command Examples:
    # Run with prompts from a JSON file, 2 parallel instances, and save outputs
    mpirun -np 3 python run_inference.py --promptfile data/prompts.json --num-parallel 2 --output-dir generated

    # Run with a single prompt and 1 instance
    mpirun -np 1 python run_inference.py --prompt "Generate a story about AI" --num-parallel 1 --output-dir generated

    # Run with 3 parallel instances and custom output directory
    mpirun -np 3 python run_inference.py --promptfile data/prompts.json --num-parallel 3 --output-dir results

    # Run with verbose output on rank 0
    mpirun -np 3 python run_inference.py --promptfile data/prompts.json --num-parallel 2 --output-dir generated --verbose
"""
import argparse
import os
import json
import shutil
from jet.file.utils import save_file
from jet.logger import logger
from jet.transformers.formatters import format_json
from mpi4py import MPI
from mlx_lm import load, generate


def run_inference(rank, size, prompts, num_parallel=None, output_dir=None, verbose=False):
    """Run inference for a subset of prompts, limited by num_parallel, and save responses if output_dir is specified."""
    # Default num_parallel to size if not specified
    num_parallel = min(
        num_parallel if num_parallel is not None else size, size)
    responses = []

    # Only ranks < num_parallel process prompts
    if rank < num_parallel:
        # Load model
        model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        # Process assigned prompts
        for i, prompt in enumerate(prompts):  # Process all assigned prompts
            if tokenizer.chat_template:
                messages = [{"role": "user", "content": prompt}]
                prompt_text = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True)
            else:
                prompt_text = prompt
            response = generate(
                model, tokenizer, prompt=prompt_text, verbose=(verbose and rank == 0))
            response_data = {
                "rank": rank,
                "prompt_index": rank + i * num_parallel,  # Adjust index for global ordering
                "prompt": prompt,
                "response": response
            }
            responses.append(response_data)
            # Save response to file if output_dir is provided
            if output_dir:
                output_file = os.path.join(
                    output_dir, f"response_rank{rank}_prompt{i}.json")
                try:
                    save_file(response_data, output_file)
                except OSError as e:
                    print(f"Error saving response to {output_file}: {e}")
    return responses


def load_prompts_from_file(promptfile):
    """Load a list of prompt strings from a JSON file."""
    if not os.path.exists(promptfile):
        raise FileNotFoundError(f"Prompt file '{promptfile}' does not exist")
    with open(promptfile, "r") as f:
        prompts = json.load(f)
    if not isinstance(prompts, list):
        raise ValueError(
            f"Prompt file '{promptfile}' must contain a JSON list")
    if not all(isinstance(p, str) for p in prompts):
        raise ValueError(f"All prompts in '{promptfile}' must be strings")
    if not prompts:
        raise ValueError(f"Prompt file '{promptfile}' is empty")
    return prompts


if __name__ == "__main__":
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    logger.orange(f"Rank: {rank}")
    logger.orange(f"Size: {size}")

    parser = argparse.ArgumentParser(
        description="Run parallel inference with MLX and mpi4py")
    parser.add_argument("--prompt", default="Hello, world!",
                        help="Single prompt for inference")
    parser.add_argument("--promptfile", default="data/prompts.json",
                        help="JSON file containing a list of prompts")
    parser.add_argument("--num-parallel", type=int, default=2,
                        help="Number of parallel model instances (defaults to MPI size)")
    parser.add_argument("--output-dir", default="generated",
                        help="Directory to save response JSON files")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output for generation")
    args = parser.parse_args()

    if args.output_dir:
        shutil.rmtree(args.output_dir, ignore_errors=True)
        os.makedirs(args.output_dir, exist_ok=True)

    # Load prompts on rank 0 and scatter to all ranks
    try:
        prompts = load_prompts_from_file(
            args.promptfile) if args.promptfile else [args.prompt]
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"Error loading prompt file: {e}")
        comm.Abort(1)

    # Distribute prompts evenly across num_parallel ranks
    num_parallel = min(args.num_parallel, size)
    if rank == 0:
        # Divide prompts into num_parallel chunks
        chunk_size = (len(prompts) + num_parallel - 1) // num_parallel
        prompt_chunks = [prompts[i * chunk_size:(i + 1) * chunk_size]
                         for i in range(num_parallel)]
        # Pad with empty lists for remaining ranks
        prompt_chunks.extend([[] for _ in range(num_parallel, size)])
    else:
        prompt_chunks = None

    # Scatter prompts to all ranks
    prompts_per_rank = comm.scatter(prompt_chunks, root=0)

    # Run inference
    responses = run_inference(
        rank, size, prompts_per_rank, num_parallel, args.output_dir, args.verbose)

    # Gather responses on rank 0 for console output
    all_responses = comm.gather(responses, root=0)

    if rank == 0:
        # Flatten responses
        flat_responses = [
            r for rank_responses in all_responses for r in rank_responses]
        # Sort by prompt_index to maintain original prompt order
        flat_responses.sort(key=lambda x: x["prompt_index"])
        # Print response texts
        print(f"Responses ({len(flat_responses)})")
        # Save to responses.json
        if args.output_dir:
            output_file = os.path.join(args.output_dir, "responses.json")
            try:
                save_file(flat_responses, output_file)
                print(
                    f"Saved JSON data {len(flat_responses)} to: {output_file}")
            except OSError as e:
                print(f"Error saving responses to {output_file}: {e}")
