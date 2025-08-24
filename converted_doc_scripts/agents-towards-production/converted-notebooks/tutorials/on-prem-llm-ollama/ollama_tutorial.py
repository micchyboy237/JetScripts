from jet.llm.mlx.base_langchain import ChatMLX
from jet.logger import CustomLogger
from langchain_community.chat_models import ChatOllama
from openai import MLX
import os
import requests
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
![](https://europe-west1-atp-views-tracker.cloudfunctions.net/working-analytics?notebook=tutorials--on-prem-llm-ollama--ollama-tutorial)

# On‑Prem Large Language Models with **Ollama**

Welcome! This tutorial shows how to **run state‑of‑the‑art language models entirely on your own hardware** using [Ollama](https://ollama.com).

**Replace External LLM APIs with Ollama!**

Learn how to replace cloud-based LLMs (MLX, Anthropic, etc.) with local Ollama models in your AI agents. This tutorial covers everything from basic API calls to complete agent migration.

Ollama is perfect for getting started with local LLMs, though advanced users may later explore alternatives like vLLM, TensorRT-LLM, or custom inference servers for maximum performance.

---

## Overview  
**Ollama** is a lightweight runtime that lets you download, run, and interact with open‑weight LLMs (like Llama 3) through a local REST API. There is **no external cloud** involvement and your data never leaves the machine.

| Section | What you’ll learn |
|---------|-------------------|
| 1. Installation | Set up the Ollama daemon on macOS / Linux / Windows or Docker |
| 2. Model Download | Pull a quantised model file (`.gguf`) and start the server |
| 3. Test Your Environment | Send a chat request with `curl` and inspect the streaming response |
| 4. Python API | Call the REST endpoint from `requests` and **LangChain’s** `ChatOllama` wrapper |
| 5. API Parameters | Customize model behavior with parameters like temperature, top_p, and system prompt |
| 6. Troubleshooting | Solve common issues |
| 7. Practice Ollama Use Cases | Example guides |

---

## Motivation  
Many organisations need generative‑AI capabilities **without sending sensitive IP to external services**. Running models on‑prem offers:

* **Data sovereignty** – full control over data and model weights  
* **Predictable costs** – no per‑token fees, just hardware utilisation  
* **Low latency** – everything happens on LAN speeds  
* **Flexibility** – swap models, tweak quantisation, or fine‑tune offline

---

## Key Components  

1. **Ollama Daemon** – background process that loads the model and exposes a REST/GRPC interface.  
2. **Model Files** (`.gguf`) – quantised weights optimised for consumer‑grade GPUs/CPUs.  
3. **Client Interfaces** – CLI (`ollama run ...`), REST, Python SDK, LangChain integration.

---

## Method Details  

We will walk through:

1. **Installation** – one‑line script or Docker image.  
2. **Model Pull & Serve** – grab *Llama 3.1 8‑b* and launch the runtime.  
3. **First Chat Request** – send a `curl` call and watch the JSON stream.  
4. **Python Requests Client** – wrap the call in pure Python.  
5. **LangChain Integration** – drop‑in replacement for cloud models.  

---

## Benefits 

* ✅ Keeps confidential data on-premises, making it suitable for enterprise environments and sensitive workloads
* ✅ Operates fully offline or in air-gapped systems, which is essential for secure or disconnected use cases
* ✅ Simple to install and run, with no need for complex setup or reliance on cloud services
* ✅ Easily integrates with existing tools, including CLI, REST API, LangChain, and others
* ✅ Eliminates per-token charges and API limits, offering a cost-effective solution for large-scale or continuous use
* ✅ Gives you full control over models and configurations, allowing customization for performance, size, and behavior

> **Is Ollama relevant for you?**  
> Ollama is especially useful when you need to run large language models on-premises for privacy, security, regulatory compliance, or to avoid cloud dependencies. Whether you're building internal tools, developing AI agents for sensitive workflows, or operating in air-gapped environments, Ollama delivers a flexible and reliable local LLM solution.

---

## Quick Start

> **📝 Note:** All examples in this tutorial use `stream: false` for simplicity and easier response handling. You can check out the `stream: true` to receive responses as they're generated in real-time, which is great for interactive applications but requires handling partial responses.

### Minimal Hardware Requirements

- **RAM**: 8GB minimum (16GB+ recommended for larger models)
- **Storage**: 10GB+ free space (models range from 4GB to 70GB+)
- **CPU**: Any modern x64 processor (ARM64 also supported on macOS)
- **GPU**: Optional but recommended (NVIDIA, AMD, or Apple Silicon for acceleration)

---

*Run each cell with* <kbd>Shift</kbd> + <kbd>Enter</kbd>, *or use the “Run All” button in the toolbar.*

# 1. Installation
Run the following command to download and install the Ollama daemon for macOS / Linux. Windows users can grab the `.exe` from the official site.

# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download .exe file from https://ollama.com/download and install it. Note, Ollama will lunch directly after installing, no need for further action.

> **💡 Tip**: You can also run Ollama using Docker containers! See the [official Docker image guide](https://ollama.com/blog/ollama-is-now-available-as-an-official-docker-image) for setup instructions.

# 2. Download a pre‑trained model & start the server
`ollama pull` fetches the quantised weights. `ollama serve` launches the local REST service.

ollama pull llama3.1:8b  # For all avilable models visit: https://ollama.com/library
ollama serve             # Start the service

> **⚠️ Windows Note**: If you get "only one usage of each socket address is permitted", Ollama is likely already running. On Windows, Ollama typically starts automatically after installation. Try skipping `ollama serve` and go directly to testing.
>

# 3. Test Your Environmen
Make your first chat request via REST API

This `curl` call sends a simple message to the local server and streams the response back as JSON.

**Linux / IOS command:**

curl http://localhost:11434/api/chat -d '{
  "model": "llama3.1:8b",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": false
}'

**Windows CMD alternative:**

curl -X POST http://localhost:11434/api/chat  -H "Content-Type: application/json"  -d "{\"model\": \"llama3.1:8b\", \"stream\": false, \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}"

**Postman alternative:**
- **Method**: POST
- **URL**: `http://localhost:11434/api/chat`
- **Headers**: `Content-Type: application/json`
- **Body** (raw JSON):
{
  "model": "llama3.1:8b",
  "stream": false,
  "messages": [
    {
      "role": "user", 
      "content": "Hello!"
    }
  ]
}

# 4. Python API

### Replace MLX API Calls

**Before (MLX):**
"""
logger.info("# On‑Prem Large Language Models with **Ollama**")


prompt = "Hello!"
client = MLX(api_key="your-key")
response = client.chat.completions.create(
    model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats",
    messages=[{"role": "user", "content": prompt}]
)
logger.debug(response.choices[0].message.content)

"""
**After (Ollama):**
"""


prompt = "Hello!"
response = requests.post("http://localhost:11434/api/chat", json={
    "model": "llama3.1:8b",
    "messages": [{"role": "user", "content": prompt}],
    "stream": False
})
data = response.json()
logger.debug(data["message"]["content"])

"""
### Replace LangChain Models

**Before (MLX):**
"""
logger.info("### Replace LangChain Models")


llm = ChatMLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats")
response = llm.invoke("Hello!")

"""
**After (Ollama):**
"""


llm = ChatOllama(model="qwen3-1.7b-4bit")
response = llm.invoke("Hello!")
logger.debug(response.content)

"""
# 5. Ollama API Parameters

Ollama provides extensive customization options. Here are the most commonly used parameters:

### Essential Parameters

| **Parameter**    | **Type** | **Default** | **Description**                                                                              | **When to Use & How**                                                                                                                                                 |
| ---------------- | -------- | ----------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`          | string   | –           | **Required.** Model identifier (e.g., `"llama3.1:8b"`).                                      | Always required. Choose the model based on your task (e.g., use `llama3.1:8b` for fast and light response or `llama3.1:70b` for higher quality if resources allow). |
| `messages`       | array    | –           | Chat history as a list of `{role: user/assistant/system, content: ...}` messages.            | Required for context. |
| `stream`         | boolean  | `true`      | Whether to stream the response token-by-token.                                               | Use `true` for real-time UI or chat apps. Use `false` if you need to process the whole response at once.|
| `temperature`    | float    | `0.8`       | Controls randomness. `0.0` = deterministic, `2.0` = very random.                             |Lower (`0.0–0.3`) to reduce variability and make outputs more predictable. Increase (`>1.0`) when diverse or exploratory outputs are acceptable.|
| `top_p`          | float    | `0.9`       | Nucleus sampling: only sample from top tokens whose cumulative probability is `top_p`.       | Lower (`0.5–0.8`) for conservative answers; keep `0.9` for balanced output. Use with or instead of `temperature`.                                                     |
| `num_predict`    | int      | `128`       | Maximum number of tokens to generate. `-1` means unlimited (until `stop` or internal limit). | Increase for longer answers. Use smaller values (`50–150`) for concise answers, larger (`256+`) for essays. Limit to control latency.                                 |
| `repeat_penalty` | float    | `1.1`       | Penalizes repetition. `1.0` = no penalty, >1 discourages repeated phrases.                   | Increase to `1.2–1.5` if the model repeats content. Keep at `1.0–1.1` for natural repetition (like poetry).                                                           |
| `system`         | string   | –           | Optional system prompt to set the behavior of the assistant.                                 | Use to define tone, expertise, or task. Example: `"You are a helpful medical assistant."`                                                                             |
| `stop`           | array    | –           | List of strings that will stop generation when encountered.                                  | Use to prevent run-on responses. Example: `["\nUser:", "</s>"]` or custom delimiters for tools.                                                                       |


### Performance Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_ctx` | int | 2048 | Context window size (max tokens in memory) |
| `num_gpu` | int | -1 | GPU layers (-1=auto, 0=CPU only) |
| `keep_alive` | string | - | Keep model loaded ("5m", "10s", "-1"=forever) |

### Usage Examples

**Basic API call with common parameters:**
"""
logger.info("# 5. Ollama API Parameters")


response = requests.post("http://localhost:11434/api/chat", json={
	"model": "llama3.1:8b",
	"messages": [{"role": "user", "content": "Explain quantum computing"}],
	"stream": False,
	"temperature": 0.3,  # Lower for factual responses
	"top_p": 0.9,  # Nucleus sampling
	"num_predict": 500,  # Limit response length
	"repeat_penalty": 1.2,  # Reduce repetition
	"stop": ["```", "---"]  # Stop at code blocks or separators
})
data = response.json()

logger.debug(data["message"]["content"])

"""
**Using with LangChain:**
"""


llm = ChatOllama(
	model="llama3.1:8b",
	temperature=0.7,
	top_p=0.9,
	num_predict=256,
	repeat_penalty=1.1
)
response = llm.invoke("Hello!")
logger.debug(response.content)

"""
> **💡 Tips:** Use `stream: false` for simpler processing • Set `num_predict` to limit response length • Use `keep_alive` to avoid reloading models • Adjust `temperature` for creativity vs consistency

> **⚠️ Note:** Not all parameters work with every model. Large `num_ctx` values require more RAM/VRAM.

# 6. Troubleshooting

**Model not found?**

ollama pull <model-name>

**Connection refused?**

ollama serve

**Out of memory?**
Try a smaller model like `mistral:7b`

### Model Recommendations

| Model | Size | Best For | Speed |
|-------|------|----------|-------|
| `llama3.1:8b` | 8GB RAM | General use, agents | Fast |
| `qwen2.5:14b` | 14GB RAM | Code, reasoning | Medium |
| `phi3:14b` | 14GB RAM | Efficient tasks | Fast |
| `mistral:7b` | 7GB RAM | Simple tasks | Very Fast |

# 7. Practice Ollama Use Cases

- Start with `basic_usage.ipynb` for simple replacements
- Try `langchain_agent.ipynb` for agent patterns  
- Experiment with different models for your use case

Ready to make your agents fully on-premises? Start with the examples!
"""
logger.info("# 6. Troubleshooting")

logger.info("\n\n[DONE]", bright=True)