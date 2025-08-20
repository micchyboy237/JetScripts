import os
import shutil
from jet.llm.mlx.base import MLX
from jet.llm.mlx.generation import chat
from jet.llm.mlx.mlx_types import ChatTemplateArgs
from jet.models.model_types import LLMModelType
from jet.file.utils import save_file
from jet.transformers.formatters import format_json
from jet.logger import logger


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), "generated", os.path.splitext(os.path.basename(__file__))[0])

MLX_LOG_DIR = f"{OUTPUT_DIR}/mlx-logs"

model: LLMModelType = "qwen3-1.7b-4bit"
seed = 42


messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Tell me a very short story about a brave knight."},
]

logger.debug("Streaming Chat Response:")
response = ""


DEFAULT_CHAT_TEMPLATE_ARGS: ChatTemplateArgs = {
    "add_generation_prompt": True,
    "tokenize": True,
    "add_special_tokens": True,
    "truncation": True,
    "max_length": 2048,
    "include_system_prompt": True,
    "tool_choice": "auto",
    "enable_thinking": False,
}


client = MLX(seed=seed, chat_template_args=DEFAULT_CHAT_TEMPLATE_ARGS)

response = chat(
    messages=messages,
    model=model,
    # max_tokens=200,
    temperature=0.3,
    log_dir=MLX_LOG_DIR,
    verbose=True,
    client=client,
    chat_template_args={
        "enable_thinking": True,
    }
)

messages.append(
    {"role": "assistant",
        "content": response["content"]}
)

save_file(messages, f"{OUTPUT_DIR}/chat.json")
