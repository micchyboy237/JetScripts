from jet.logger import logger
from langchain_community.chat_models.llama_edge import LlamaEdgeChatService
from langchain_core.messages import HumanMessage, SystemMessage
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# LlamaEdge

[LlamaEdge](https://github.com/second-state/LlamaEdge) allows you to chat with LLMs of [GGUF](https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/README.md) format both locally and via chat service.

- `LlamaEdgeChatService` provides developers an Ollama API compatible service to chat with LLMs via HTTP requests.

- `LlamaEdgeChatLocal` enables developers to chat with LLMs locally (coming soon).

Both `LlamaEdgeChatService` and `LlamaEdgeChatLocal` run on the infrastructure driven by [WasmEdge Runtime](https://wasmedge.org/), which provides a lightweight and portable WebAssembly container environment for LLM inference tasks.

## Chat via API Service

`LlamaEdgeChatService` works on the `llama-api-server`. Following the steps in [llama-api-server quick-start](https://github.com/second-state/llama-utils/tree/main/api-server#readme), you can host your own API service so that you can chat with any models you like on any device you have anywhere as long as the internet is available.
"""
logger.info("# LlamaEdge")


"""
### Chat with LLMs in the non-streaming mode
"""
logger.info("### Chat with LLMs in the non-streaming mode")

service_url = "https://b008-54-186-154-209.ngrok-free.app"

chat = LlamaEdgeChatService(service_url=service_url)

system_message = SystemMessage(content="You are an AI assistant")
user_message = HumanMessage(content="What is the capital of France?")
messages = [system_message, user_message]

response = chat.invoke(messages)

logger.debug(f"[Bot] {response.content}")

"""
### Chat with LLMs in the streaming mode
"""
logger.info("### Chat with LLMs in the streaming mode")

service_url = "https://b008-54-186-154-209.ngrok-free.app"

chat = LlamaEdgeChatService(service_url=service_url, streaming=True)

system_message = SystemMessage(content="You are an AI assistant")
user_message = HumanMessage(content="What is the capital of Norway?")
messages = [
    system_message,
    user_message,
]

output = ""
for chunk in chat.stream(messages):
    output += chunk.content

logger.debug(f"[Bot] {output}")

logger.info("\n\n[DONE]", bright=True)