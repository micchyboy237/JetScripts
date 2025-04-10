from jet.llm.ollama.base import Ollama
from jet.logger import logger
from jet.transformers.formatters import format_json

prompt = """You are a helpful assistant that assigns concise and meaningful labels to text for classification purposes.

Text:
"Deploy the mobile application to both App Store and Google Play."

Generate 1 to 3 short, high-level labels that best describe the main topics or intent of this text.
Return them as a comma-separated list.
Labels:"""

llm = Ollama(model="gemma3:1b")
response = llm.chat(prompt)

logger.success(format_json(response))
