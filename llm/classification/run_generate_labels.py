from jet.llm.mlx.base import MLX
from jet.llm.mlx.mlx_types import ModelKey
from jet.llm.ollama.base import Ollama
from jet.logger import logger
from jet.transformers.formatters import format_json

prompt = """You are a helpful assistant that assigns concise and meaningful labels to text for classification purposes.

Text:
"Deploy the mobile application to both App Store and Google Play."

Generate 1 to 3 short, high-level labels that best describe the main topics or intent of this text.
Return them as a comma-separated list.
Labels:"""

model: ModelKey = "llama-3.2-1b-instruct-4bit"

mlx = MLX(model)
response_stream = mlx.stream_chat(prompt)
response = ""
for chunk in response_stream:
    content = chunk["choices"][0]["message"]["content"]
    response += content
    logger.success(content, flush=True)

logger.success(format_json(response))
