from jet.data.utils import generate_unique_hash
from jet.llm.ollama.base import Ollama
import time

from jet.logger import logger
from jet.transformers.formatters import format_json
from llama_index.core.base.llms.types import ChatMessage

system = "You are a helpful AI assistant."

# Initialize Ollama with a unique session ID
session_id = generate_unique_hash()
ollama = Ollama(model="llama3.1", session_id=session_id)

# First chat interaction
response1 = ollama.chat("Hey, who won the 2024 NBA finals?", system=system)
logger.debug("Response 1:", response1.message.content)

history = ollama.chat_history.get_messages()
logger.success(format_json(history))
logger.info(f"History 1 length: {len(history)}")

# Wait a bit then ask a follow-up (relies on history)
time.sleep(2)
response2 = ollama.chat("Cool. Who was MVP?", system=system)
logger.debug("Response 2:", response2.message.content)

history = ollama.chat_history.get_messages()
logger.success(format_json(history))
logger.info(f"History 2 length: {len(history)}")

# Another way: provide a list of ChatMessage for multistep conversation

messages = [
    ChatMessage(role="system", content=system),
    ChatMessage(role="user", content="Tell me a joke about space."),
    ChatMessage(role="assistant",
                content="Why don’t astronauts get hungry after being blasted into space? Because they’ve just had a big launch!"),
    ChatMessage(role="user", content="Another one?")
]

response3 = ollama.chat(messages, system=system)
logger.debug("Response 3:", response3.message.content)

# View conversation history from DB (via PostgresChatMessageHistory)
history = ollama.chat_history.get_messages()
logger.success(format_json(history))
logger.info(f"History 3 length: {len(history)}")
