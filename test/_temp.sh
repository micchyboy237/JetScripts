# llama.cpp
llama-server -hf ggml-org/gemma-3-4b-it-GGUF --host 0.0.0.0 --port 8080
llama-server -hf bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:Q4_K_M --host 0.0.0.0 --port 8080
llama-server -hf ggml-org/Qwen2.5-VL-7B-Instruct-GGUF:Q4_K_M --host 0.0.0.0 --port 8080
llama-server --jinja -fa on -hf bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M --host 0.0.0.0 --port 8080
llama-server --jinja -fa on -hf bartowski/Mistral-Nemo-Instruct-2407-GGUF:Q4_K_M --host 0.0.0.0 --port 8080
llama-server --jinja -fa on -hf bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M --chat-template-file models/templates/llama-cpp-deepseek-r1.jinja --host 0.0.0.0 --port 8080
llama-server --jinja -fa on -hf bartowski/Hermes-3-Llama-3.1-8B-GGUF:Q4_K_M --chat-template-file models/templates/NousResearch-Hermes-3-Llama-3.1-8B-tool_use.jinja --host 0.0.0.0 --port 8080

# Chat CURL command
curl -N http://shawn-pc.local:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain black holes in simple terms."}
    ],
    "stream": true
  }'
