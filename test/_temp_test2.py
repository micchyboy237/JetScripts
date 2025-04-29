"""
Run parallel inference with the MLX framework for the Llama-3.2-1B-Instruct-4bit model.

This script supports generating responses from a list of prompts (via --promptfile or --prompt),
limiting parallel model instances (--num-parallel), generating a hostfile (--generate-hostfile),
and saving responses to a directory (--output-dir).

Command Examples:
    # Run standalone with prompts from a JSON file, 2 parallel instances, and save outputs
    python run_inference.py --generate-hostfile --promptfile data/prompts.json --num-parallel 2 --output-dir generated

    # Run with distributed_run for distributed processing
    python distributed_run.py --hostfile data/hosts.txt -- python run_inference.py --promptfile data/prompts.json --num-parallel 2 --output-dir generated

    # Run with a single prompt
    python run_inference.py --generate-hostfile --prompt "Generate a story about AI" --num-parallel 1 --output-dir generated

    # Customize slots and parallel instances
    python run_inference.py --generate-hostfile --slots 4 --promptfile data/prompts.json --num-parallel 3 --output-dir generated

    # Use hostname instead of IP in hostfile
    python run_inference.py --generate-hostfile --use-ip=False --promptfile data/prompts.json --num-parallel 2 --output-dir generated
"""
import socket
from mlx_lm import load, generate
import argparse
import os
import json


def run_inference(rank, hostfile, prompts, num_parallel=None, output_dir=None):
    """Run inference for a subset of prompts, limited by num_parallel, and save responses if output_dir is specified."""
    model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    responses = []
    num_ranks = int(os.environ.get("MLX_WORLD_SIZE", "1"))
    # Default num_parallel to num_ranks if not specified
    num_parallel = num_parallel if num_parallel is not None else num_ranks
    # Only process prompts if rank is within num_parallel limit
    if rank < num_parallel:
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        for i, prompt in enumerate(prompts[rank::num_parallel]):
            if tokenizer.chat_template:
                messages = [{"role": "user", "content": prompt}]
                prompt_text = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True)
            else:
                prompt_text = prompt
            response = generate(
                model, tokenizer, prompt=prompt_text, verbose=(rank == 0))
            response_data = {
                "rank": rank,
                "prompt_index": rank + i * num_parallel,
                "prompt": prompt,
                "response": response
            }
            responses.append(response_data)
            # Save response to file if output_dir is provided
            if output_dir:
                output_file = os.path.join(
                    output_dir, f"response_rank{rank}_prompt{i}.json")
                try:
                    with open(output_file, "w") as f:
                        json.dump(response_data, f, indent=2)
                except OSError as e:
                    print(f"Error saving response to {output_file}: {e}")
    return responses


def resolve_hostname(hostname):
    """Resolve hostname to IP address."""
    try:
        ip_address = socket.gethostbyname(hostname)
        return ip_address
    except socket.gaierror as e:
        raise ValueError(f"Failed to resolve hostname '{hostname}': {e}")


def generate_hosts_file(hostname, slots, use_ip=True, output_file="hosts.txt"):
    """Generate hosts.txt with the specified hostname or IP and number of slots."""
    if use_ip:
        host = resolve_hostname(hostname)
    else:
        host = hostname
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write(f"{host} slots={slots}\n")
    print(f"Generated {output_file}: {host} slots={slots}")


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
    parser = argparse.ArgumentParser(
        description="Run parallel inference with MLX")
    parser.add_argument(
        "--hostfile", default="data/hosts.txt", help="Path to hostfile")
    parser.add_argument("--prompt", default="Hello, world!",
                        help="Single prompt for inference")
    parser.add_argument("--promptfile", default="data/prompts.json",
                        help="JSON file containing a list of prompts")
    parser.add_argument("--num-parallel", type=int, default=2,
                        help="Number of parallel model instances (defaults to MLX_WORLD_SIZE)")
    parser.add_argument("--output-dir", default="generated",
                        help="Directory to save response JSON files")
    parser.add_argument("--generate-hostfile", action="store_true", default=True,
                        help="Generate hostfile before running inference")
    parser.add_argument("--hostname", default="Jethros-MacBook-Air.local",
                        help="Hostname for hostfile generation")
    parser.add_argument("--slots", type=int, default=3,
                        help="Number of slots for hostfile generation")
    parser.add_argument("--use-ip", action="store_true", default=True,
                        help="Use IP address instead of hostname in hostfile")
    args, rest = parser.parse_known_args()

    # Generate hostfile if requested
    if args.generate_hostfile:
        generate_hosts_file(args.hostname, args.slots,
                            args.use_ip, args.hostfile)

    # Load prompts
    if args.promptfile:
        try:
            prompts = load_prompts_from_file(args.promptfile)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
            print(f"Error loading prompt file: {e}")
            exit(1)
    else:
        prompts = [args.prompt] if isinstance(
            args.prompt, str) else args.prompt

    # Set MLX_RANK environment variable
    os.environ["MLX_RANK"] = os.environ.get("MLX_RANK", "0")

    # Run inference
    responses = run_inference(
        int(os.environ["MLX_RANK"]),
        args.hostfile,
        prompts,
        args.num_parallel,
        args.output_dir
    )
    if int(os.environ["MLX_RANK"]) == 0:
        # Print response texts for console output
        print("Responses:", [r["response"] for r in responses])
