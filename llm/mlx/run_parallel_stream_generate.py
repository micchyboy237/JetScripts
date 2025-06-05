from jet.llm.mlx.base import MLX
from jet.llm.mlx.mlx_types import LLMModelKey
import mlx.core as mx
import numpy as np
from mpi4py import MPI
from mlx_lm import load
from mlx_lm.generate import generate_step
from jet.logger import logger

"""
Sample Commands:
mpirun -np 4 python run_parallel_stream_generate.py
"""

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define prompts (in practice, load from a file or generate dynamically)
prompts = [
    "Write a story about Einstein",
    "Explain quantum mechanics in simple terms",
    "Generate a poem about the universe",
    "Describe a futuristic city",
    # Add more prompts as needed
]

# Distribute prompts across processes
prompts_per_process = len(prompts) // size
start_idx = rank * prompts_per_process
end_idx = (rank + 1) * prompts_per_process if rank < size - 1 else len(prompts)
local_prompts = prompts[start_idx:end_idx]

# Load the model and tokenizer (each process loads its own instance)
model_path: LLMModelKey = "llama-3.2-1b-instruct-4bit"
# model, tokenizer = load(model_path)
mlx = MLX(model_path)

# Generate text for each local prompt using streaming
local_results = []
for prompt in local_prompts:
    # if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
    #     messages = [{"role": "user", "content": prompt}]
    #     prompt = tokenizer.apply_chat_template(
    #         messages, tokenize=False, add_generation_prompt=True
    #     )

    # # Tokenize the prompt
    # prompt_tokens = tokenizer.encode(prompt)
    # prompt_array = mx.array(prompt_tokens)

    # # Stream generation
    # response_tokens = []
    # for (token, prob), step in zip(generate_step(prompt_array, model), range(512)):
    #     response_tokens.append(token)
    #     if rank == 0:  # Optionally print tokens as they are generated for rank 0
    #         logger.success(tokenizer.decode([token]), end='', flush=True)
    #     if token == tokenizer.eos_token_id:
    #         break

    # # Decode the response
    # response = tokenizer.decode(response_tokens)

    response = ""
    stream_response = mlx.stream_chat(
        prompt,
        model=model_path,
        temperature=0
    )
    for chunk in stream_response:
        content = chunk["choices"][0]["message"]["content"]
        logger.success(content, flush=True)
        response += content

    local_results.append((prompt, response))
    if rank == 0:
        print()  # Newline after streaming

# Gather results from all processes
all_results = comm.gather(local_results, root=0)

# Process results on rank 0
if rank == 0:
    # Flatten the list of results
    flattened_results = [item for sublist in all_results for item in sublist]
    for i, (prompt, response) in enumerate(flattened_results):
        print(f"Prompt {i+1}: {prompt}\nResponse: {response}\n{'-'*50}")
