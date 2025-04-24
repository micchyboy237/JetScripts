from jet.logger import logger
from jet.transformers.formatters import format_json
from llama_cpp import Llama

model_path = "/Users/jethroestrada/.cache/huggingface/hub/models--TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF/snapshots/52e7645ba7c309695bec7ac98f4f005b139cf465/tinyllama-1.1b-chat-v1.0.Q5_0.gguf"
# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = Llama(
    # Download the model file first
    model_path=model_path,
    n_ctx=2048,  # The max sequence length to use - note that longer sequence lengths require much more resources
    # The number of CPU threads to use, tailor to your system and the resulting performance
    n_threads=8,
    # The number of layers to offload to GPU, if you have GPU acceleration available
    # n_gpu_layers=35
)

# Simple inference example
output = llm(
    # Prompt
    "<|system|>\n{system_message}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>",
    max_tokens=512,  # Generate up to 512 tokens
    # Example stop token - not necessarily correct for this specific model! Please check before using.
    stop=["</s>"],
    echo=True        # Whether to echo the prompt
)

logger.gray("Generate response:")
logger.success(format_json(output))

# Chat Completion API

# Set chat_format according to the model you are using
llm = Llama(model_path=model_path,
            chat_format="llama-2")
response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a story writing assistant."},
        {
            "role": "user",
            "content": "Write a story about llamas."
        }
    ]
)

logger.gray("Chat response:")
logger.success(format_json(response))
