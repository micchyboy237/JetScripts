from jet.llm.mlx.mlx_types import ModelType
from jet.llm.mlx.base import MLX
from jet.logger import logger
from jet.transformers.formatters import format_json
from mpi4py import MPI
import numpy as np
import json
import sys


def parallel_stream_generate(
    prompts: list,
    model_name: ModelType = "llama-3.2-1b-instruct-4bit",
    max_tokens: int = 100,
    temp: float = 0.7,
    verbose: bool = False
) -> list:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    try:
        mlx = MLX(model_name)
    except Exception as e:
        print(
            f"error: Failed to load model {model_name}: {str(e)}", flush=True)
        return []

    prompts_per_process = len(prompts) // size
    start_idx = rank * prompts_per_process
    end_idx = (rank + 1) * prompts_per_process if rank < size - \
        1 else len(prompts)
    local_prompts = prompts[start_idx:end_idx]
    local_results = []

    for prompt in local_prompts:
        response = ""
        try:
            if verbose:
                print(
                    f"data: Process {rank} generating for prompt: {prompt}", flush=True)
            stream_response = mlx.stream_chat(
                prompt,
                model=model_name,
                temperature=temp,
                max_tokens=max_tokens
            )
            for chunk in stream_response:
                content = chunk["choices"][0]["message"]["content"]
                response += content
                print(f"data: Process {rank}: {content}", flush=True)
            local_results.append((prompt, response))
            if verbose and rank == 0:
                print("data: ", flush=True)
        except Exception as e:
            print(
                f"error: Process {rank} failed for prompt '{prompt}': {str(e)}", flush=True)

    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        try:
            flattened_results = [
                item for sublist in all_results for item in sublist]
            for i, (prompt, response) in enumerate(flattened_results):
                result = json.dumps({"prompt": prompt, "response": response})
                print(f"result: Result {i+1}: {result}", flush=True)
            return flattened_results
        except Exception as e:
            print(f"error: Failed to process results: {str(e)}", flush=True)

    return []


if __name__ == "__main__":
    logger.debug(format_json(sys.argv))
    if len(sys.argv) < 2:
        print("error: Usage: mpirun -np 4 python _test_for_running_temp_scripts.py <input_json>", flush=True)
        sys.exit(1)

    try:
        # Join all arguments after script name to handle JSON split across args
        input_json = ' '.join(sys.argv[1:]).strip("'")
        input_data = json.loads(input_json)
    except json.JSONDecodeError as e:
        print(f"error: Invalid JSON input: {str(e)}", flush=True)
        sys.exit(1)

    results = parallel_stream_generate(
        model_name=input_data["model"],
        prompts=input_data["prompts"],
        max_tokens=input_data["max_tokens"],
        temp=input_data["temp"],
        verbose=input_data["verbose"]
    )
    MPI.Finalize()
