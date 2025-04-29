#!/bin/zsh

# Set environment variable for efficient tokenization
export TOKENIZERS_PARALLELISM=true

# Install required packages (uncomment if not already installed)
# pip install mlx-lm datasets

# Create directory for the sample dataset
mkdir -p data/sample_chat

# Write train.jsonl
cat << EOF > data/sample_chat/train.jsonl
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "The capital of France is Paris."}]}
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "How do I make a simple salad?"}, {"role": "assistant", "content": "Chop lettuce, tomatoes, and cucumbers. Toss with olive oil, lemon juice, salt, and pepper."}]}
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is 2 + 2?"}, {"role": "assistant", "content": "2 + 2 equals 4."}]}
EOF

# Write valid.jsonl
cat << EOF > data/sample_chat/valid.jsonl
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the largest planet in our solar system?"}, {"role": "assistant", "content": "The largest planet is Jupiter."}]}
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "How do I boil an egg?"}, {"role": "assistant", "content": "Place egg in a pot of water, bring to a boil, then simmer for 10 minutes."}]}
EOF

# Write test.jsonl
cat << EOF > data/sample_chat/test.jsonl
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the boiling point of water?"}, {"role": "assistant", "content": "The boiling point of water is 100°C at standard pressure."}]}
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the capital of Japan?"}, {"role": "assistant", "content": "The capital of Japan is Tokyo."}]}
EOF

# Ensure Hugging Face authentication (uncomment and set token if needed)
# huggingface-cli login --token your_hf_token

# Optional: Quantize the model for QLoRA (uncomment to run)
# python convert.py -q --model meta-llama/Llama-3.2-1B-Instruct

# Fine-tune the model
mlx_lm.lora \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --train \
    --data data/sample_chat \
    --batch-size 1 \
    --num-layers 4 \
    --iters 100 \
    --learning-rate 1e-5 \
    --adapter-path adapters \
    --grad-checkpoint \
    --mask-prompt \
    --val-batches 2 \
    --steps-per-eval 50 \
    --save-every 50

# Evaluate the model
mlx_lm.lora \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --adapter-path adapters \
    --data data/sample_chat \
    --test \
    --test-batches 2

# Generate output with a sample prompt
mlx_lm.generate \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --adapter-path adapters \
    --prompt "You are a helpful assistant. What is the capital of Brazil?"

# Optional: Fuse the model (uncomment to run)
# mlx_lm.fuse --model meta-llama/Llama-3.2-1B-Instruct

# Optional: Fuse and upload to Hugging Face (uncomment to run)
# mlx_lm.fuse \
#     --model meta-llama/Llama-3.2-1B-Instruct \
#     --upload-repo mlx-community/my-lora-llama-3.2-1b \
#     --hf-path meta-llama/Llama-3.2-1B-Instruct

# Optional: Export to GGUF (uncomment to run)
# mlx_lm.fuse \
#     --model meta-llama/Llama-3.2-1B-Instruct \
#     --export-gguf