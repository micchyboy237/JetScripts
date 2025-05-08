from jet.llm.mlx.mlx_types import ModelType
from jet.llm.mlx.base import MLX
from jet.logger import logger
from jet.transformers.formatters import format_json
from mpi4py import MPI
import numpy as np
import json
import sys
import uuid


def parallel_stream_generate(
    prompts: list,
    model_name: ModelType = "llama-3.2-1b-instruct-4bit",
    max_tokens: int = 100,
    temp: float = 0.7,
    verbose: bool = False,
    task_id: str = None
) -> list:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Debug: Confirm process initialization
    print(f"data: Process {rank} of {size} initialized", flush=True)

    try:
        mlx = MLX(model_name)
    except Exception as e:
        print(
            f"error: Failed to load model {model_name}: {str(e)}", flush=True)
        return []

    # Distribute prompts across processes, ensuring at least one prompt per process if possible
    local_prompts = []
    if prompts:
        # Assign prompts using a round-robin approach to handle cases where len(prompts) < size
        for i in range(len(prompts)):
            if i % size == rank:
                local_prompts.append(prompts[i])

    # Debug: Log assigned prompts
    print(
        f"data: Process {rank} assigned prompts: {local_prompts}", flush=True)

    local_results = []
    for prompt in local_prompts:
        response = ""
        prompt_id = str(uuid.uuid4())  # Unique ID for each prompt
        try:
            if verbose:
                print(
                    f"data: Process {rank} (Prompt ID: {prompt_id}) generating for prompt: {prompt}", flush=True)
            stream_response = mlx.stream_chat(
                prompt,
                model=model_name,
                temperature=temp,
                max_tokens=max_tokens
            )
            for chunk in stream_response:
                content = chunk["choices"][0]["message"]["content"]
                response += content
                print(
                    f"data: Process {rank} (Prompt ID: {prompt_id}): {content}", flush=True)
            local_results.append((prompt, response, prompt_id))
            if verbose and rank == 0:
                print("data: ", flush=True)
        except Exception as e:
            print(
                f"error: Process {rank} (Prompt ID: {prompt_id}) failed for prompt '{prompt}': {str(e)}", flush=True)

    # Gather results from all processes
    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        try:
            flattened_results = [
                item for sublist in all_results for item in sublist]
            for i, (prompt, response, prompt_id) in enumerate(flattened_results):
                result = json.dumps({
                    "prompt": prompt,
                    "response": response,
                    "prompt_id": prompt_id,
                    "task_id": task_id
                })
                print(f"result: {result}", flush=True)
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
        verbose=input_data["verbose"],
        task_id=input_data.get("task_id")
    )
    MPI.Finalize()
