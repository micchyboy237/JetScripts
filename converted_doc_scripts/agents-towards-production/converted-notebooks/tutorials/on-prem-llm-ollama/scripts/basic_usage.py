from jet.logger import logger
from langchain_community.chat_models import ChatOllama
import os
import requests
import shutil
import time


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
# Basic Ollama Usage Examples

This shows the simplest ways to replace Ollama/other API calls with Ollama.
Perfect for understanding the core concepts before moving to agents.

1) Direct API call
2) Ollama API parameters
3) Basic LangChain Integration
4) Conversation
"""
logger.info("# Basic Ollama Usage Examples")


"""
# Lets check if Ollama is runninig:
"""
logger.info("# Lets check if Ollama is runninig:")

def check_ollama_running():
    """Check if Ollama is available."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

if not check_ollama_running():
    logger.debug("❌ Ollama not running!")
    logger.debug("Start it with: ollama serve")
    logger.debug("And make sure you have a model: ollama pull llama3.1:8b")
else:
    logger.debug("✅ Ollama is running\n")
    logger.debug("🚀 Basic Ollama Usage Examples")
    logger.debug("=" * 40)

"""
# Direct API

# system vs user call:
✅ role: "system"
Purpose: Sets initial behavior, tone, or constraints for the model.
Example Use: Define the assistant's personality, scope, or instructions before the conversation starts.
The model reads it once and uses it to shape all following responses.
Not shown to the user in most applications.

✅ role: "user"
Purpose: Represents the actual question or input from the user.
Triggers the model to respond.
Can be followed by assistant messages to continue the back-and-forth dialogue.
"""
logger.info("# Direct API")

def example_1_direct_api(sys_promt, usr_promt):
    response = requests.post("http://localhost:11434/api/chat", json={
        "model": "llama3.1:8b",
        "messages": [{"role": "system", "content": sys_promt},
                     {"role": "user", "content":  usr_promt}
                     ],
        "stream": False  # Ensure it's not streaming
    })
    if response.status_code == 200:
        result = response.json()
        logger.debug(f"Response: {result["message"]["content"]}")
    else:
        logger.debug(f"Error: {response.status_code}")

logger.debug("Example 1: Direct API Call")
logger.debug("-" * 30)
sys_promt = "You are a helpful assistant."
usr_promt = "What is Python?"
example_1_direct_api(sys_promt,usr_promt)

"""
## Ollama API Parameters

Ollama provides extensive customization options. Here are the most commonly used parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | - | **Required.** Model identifier (e.g., `"llama3.1:8b"`) |
| `messages` | array | - | Chat conversation for `/api/chat` |
| `stream` | boolean | true | Stream response as it's generated |
| `temperature` | float | 0.8 | Randomness (0.0 = deterministic, 2.0 = very random) |
| `top_p` | float | 0.9 | Nucleus sampling (0.1 = top 10% tokens only) |
| `num_predict` | int | 128 | Maximum tokens to generate (-1 = unlimited) |
| `repeat_penalty` | float | 1.1 | Penalty for repeating tokens |
| `system` | string | - | System message to guide behavior |
| `stop` | array | - | Stop generation at these strings |
"""
logger.info("## Ollama API Parameters")

def example_2_with_parameters(temperature, num_predict, usr_promt):

    response = requests.post("http://localhost:11434/api/chat", json={
        "model": "llama3.1:8b",
        "messages": [{"role": "user", "content": usr_promt}],
        "temperature": temperature,  # Higher temperature for creativity
        "num_predict": num_predict,   # Limit response length
        "stream": False  # Ensure it's not streaming
    })

    if response.status_code == 200:
        result = response.json()["message"]["content"]
        logger.debug(f"Creative response: {result}")
    else:
        logger.debug(f"Error: {response.status_code}")

logger.debug("\nExample 2: With Parameters")
logger.debug("-" * 30)
usr_promt = "Write a creative story about a robot."
temperature = 0.8
num_predict = 100
example_2_with_parameters(temperature, num_predict, usr_promt)

"""
# LangChain Integration

| Feature | ChatOllama | Direct API |
|---------|------------|------------|
| `Abstraction level` | High | Low |
| `Ease of use` | Easier | Manual formatting required |
| `Customization (headers, etc.)` | Limited | Full |
| `Dependency` | Requires LangChain | Only requests | 
| `Best for` | RAG pipelines, fast prototyping | Custom tools, low-level control |
"""
logger.info("# LangChain Integration")

def example_3_langchain_basic(usr_promt):
    try:

        llm = ChatOllama(model="llama3.2")

        response = llm.invoke(usr_promt)
        logger.debug(f"LangChain response: {response.content}")

    except ImportError:
        logger.debug("LangChain not installed. Run: pip install langchain langchain-community")
    except Exception as e:
        logger.debug(f"Error: {e}")

logger.debug("\nExample 3: LangChain Integration")
logger.debug("-" * 30)
usr_promt = "Explain machine learning in one sentence."
example_3_langchain_basic(usr_promt)

"""
# Maintaining Conversation Context

In a conversation with a language model, messages are exchanged between different roles. The "assistant" role represents the model’s response to a user input. When we add {"role": "assistant", "content": assistant_msg} to the message history, we’re storing the model’s last reply. This helps maintain context in multi-turn conversations, allowing the model to remember what it said before and respond accordingly.
"""
logger.info("# Maintaining Conversation Context")

def example_4_conversation(messages):
    response = requests.post("http://localhost:11434/api/chat", json={
        "model": "llama3.1:8b",
        "messages": messages,
        "stream": False
    })

    if response.status_code == 200:
        assistant_msg = response.json()["message"]["content"]
        return assistant_msg

"""
### Start conversation:
"""
logger.info("### Start conversation:")

logger.debug("\nExample 4: Conversation")
logger.debug("-" * 30)
messages = [{"role": "system", "content": "You are a helpful coding assistant."}]
messages.append({"role": "user", "content": "How do I read a file in Python?"})
assistant_msg = example_4_conversation(messages)
logger.debug(f"Assistant:\n{assistant_msg}")

"""
### Add assistant:
"""
logger.info("### Add assistant:")

messages.append({"role": "assistant", "content": assistant_msg})

"""
### Add follow-up question:
"""
logger.info("### Add follow-up question:")

messages.append({"role": "user", "content": "What about error handling?"})
answer = example_4_conversation(messages)
logger.debug(f"\n\nFollow-up:\n{answer}")

"""
✅ All examples completed!

💡 Key Points:

• Replace Ollama URLs with http://localhost:11434

• No API keys needed!

• ChatOllama replaces ChatOllama in LangChain
"""

logger.info("\n\n[DONE]", bright=True)