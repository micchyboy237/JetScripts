# Performance Summary of GPT-2 Alternatives on Mac M1

The table below summarizes the performance of five lightweight language models for text generation on a Mac M1 (16GB RAM, 4-bit quantization for GGUF models, MPS acceleration where applicable). Metrics are based on Python implementations using Hugging Face Transformers or `llama-cpp-python`.

| Model       | Parameters | RAM Usage (4-bit) | Inference Speed (tokens/s) | Output Quality | Best Use Case               |
| ----------- | ---------- | ----------------- | -------------------------- | -------------- | --------------------------- |
| GPT-2       | 124M       | \~0.7GB           | 8-12                       | Good           |                             |
| DistilGPT-2 | 82M        | \~0.5GB           | 10-15                      | Good           | Lightweight text generation |
| TinyLlama   | 1.1B       | \~1.5GB           | 5-10                       | Very Good      | Creative writing, chatbots  |
| Grok Mini   | \~3B       | \~2-3GB           | 4-8                        | Very Good      | Dialogue, storytelling      |
| OPT-1.3B    | 1.3B       | \~1.5-2GB         | 3-7                        | Good           | General text generation     |
| Phi-3 Mini  | 3.8B       | \~2.5-3.5GB       | 5-8                        | Excellent      | Creative tasks, instruction |

## Notes

- **RAM Usage**: Estimated for 4-bit quantized GGUF models (TinyLlama, Grok Mini, Phi-3 Mini) or native models (DistilGPT-2, OPT-1.3B) on 16GB M1.
- **Inference Speed**: Measured on M1 with 16GB RAM, using MPS for Transformers models and Metal for `llama-cpp-python`. Varies with prompt length and batch size.
- **Output Quality**: Subjective assessment based on coherence, context retention, and creativity for the prompt "Once upon a time". Phi-3 Mini excels due to its larger size and instruction tuning.
- **Dependencies**: Requires `transformers`, `torch`, and `llama-cpp-python`. GGUF models need downloading from Hugging Face.
- **Hardware**: 8GB M1 sufficient for DistilGPT-2 and TinyLlama; 16GB recommended for Grok Mini, OPT-1.3B, and Phi-3 Mini.
