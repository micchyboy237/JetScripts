import os

from jet.adapters.llama_cpp.tokens import count_tokens
from jet.logger import logger

terminate_tag = "```python"
input_tokens = count_tokens(terminate_tag, model=os.getenv("LLAMA_CPP_LLM_MODEL"))
logger.info(f"Input tokens: {input_tokens}")
