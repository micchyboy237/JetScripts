import os
import shutil
from jet.llm.mlx.base import MLX
from jet.llm.mlx.mlx_types import LLMModelType
from jet.file.utils import save_file
from jet.transformers.formatters import format_json
from jet.logger import logger


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), "generated", os.path.splitext(os.path.basename(__file__))[0])

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

MLX_LOG_DIR = f"{OUTPUT_DIR}/mlx-logs"

model: LLMModelType = "llama-3.2-3b-instruct-4bit"
seed = 42


"""Example of using the .stream_chat method for streaming chat completions."""
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Tell me a very short story about a brave knight."},
]

logger.debug("Streaming Chat Response:")
response = ""

client = MLX(seed=seed)

for stream_response in client.stream_chat(
    messages=messages,
    model=model,
    # max_tokens=200,
    temperature=0.3,
    log_dir=MLX_LOG_DIR
):
    content = stream_response["choices"][0]["message"]["content"]
    response += content
    logger.success(content, flush=True)

    if stream_response["choices"][0]["finish_reason"]:
        logger.newline()

messages.append(
    {"role": "assistant", "content": response}
)

save_file(messages, f"{OUTPUT_DIR}/stream_chat.json")
