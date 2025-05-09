from jet.llm.mlx.base import MLX
from mpi4py import MPI
from mlx_lm import load, generate
import numpy as np

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
model_path = "mlx-community/Llama-3.2-1B-Instruct-4bit"
# model, tokenizer = load(model_path)
mlx = MLX(model_path)

# Generate text for each local prompt
local_results = []
for prompt in local_prompts:
    # if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
    #     messages = [{"role": "user", "content": prompt}]
    #     prompt = tokenizer.apply_chat_template(
    #         messages, tokenize=False, add_generation_prompt=True
    #     )
    # response = generate(
    #     model,
    #     tokenizer,
    #     prompt=prompt,
    #     max_tokens=512,
    #     verbose=True
    # )
    response = mlx.chat(
        prompt,
        model=model_path,
        temperature=0
    )
    content = response["choices"][0]["message"]["content"]
    local_results.append((prompt, content))

# Gather results from all processes
all_results = comm.gather(local_results, root=0)

# Process results on rank 0
if rank == 0:
    # Flatten the list of results
    flattened_results = [item for sublist in all_results for item in sublist]
    for prompt, response in flattened_results:
        print(f"Prompt: {prompt}\nResponse: {response}\n{'-'*50}")
