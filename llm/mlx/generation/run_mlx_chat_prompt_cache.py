import os
import shutil
import time
from mlx_lm import load, generate
from mlx_lm.models.cache import (
    make_prompt_cache,
    can_trim_prompt_cache,
    trim_prompt_cache,
    save_prompt_cache,
    KVCache,
    RotatingKVCache,
    QuantizedKVCache,
)


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Load model
model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

# Create a rotating cache with max size
cache = make_prompt_cache(model, max_kv_size=128)  # Uses RotatingKVCache

# Optionally quantize cache
# for i in range(len(cache)):
#     if hasattr(cache[i], "to_quantized"):
#         cache[i] = cache[i].to_quantized(group_size=64, bits=8)

# Generate with cache
prompt = "Write a Python function for sorting."
start = time.time()
generate(model, tokenizer, prompt="Summarize quantum computing",
         max_tokens=512, prompt_cache=cache, verbose=True)
print(f"Time: {time.time() - start} seconds")

# Trim cache if needed
if can_trim_prompt_cache(cache):
    trim_prompt_cache(cache, num_tokens=64)

# Save cache for reuse
save_prompt_cache(
    f"{os.path.dirname(__file__)}/prompt_cache.safetensors", cache)
