import os
import shutil
from jet.llm.mlx.base import MLX
from jet.llm.mlx.generation import stream_chat
from jet.models.model_types import LLMModelType
from jet.file.utils import save_file
from jet.transformers.formatters import format_json
from jet.logger import logger


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), "generated", os.path.splitext(os.path.basename(__file__))[0])

MLX_LOG_DIR = f"{OUTPUT_DIR}/mlx-logs"

model: LLMModelType = "llama-3.2-1b-instruct-4bit"
seed = 42


messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Tell me a very short story about a brave knight."},
]

logger.debug("Streaming Chat Response:")
response = ""

client = MLX(seed=seed)

for stream_response in stream_chat(
    messages=messages,
    model=model,
    # max_tokens=200,
    temperature=0.3,
    log_dir=MLX_LOG_DIR,
    verbose=True,
    client=client,
    logprobs=5
):
    content = stream_response["choices"][0]["message"]["content"]
    response += content

    if stream_response["choices"][0]["finish_reason"]:
        logger.newline()

messages.append(
    {"role": "assistant", "content": response}
)

save_file(messages, f"{OUTPUT_DIR}/stream_chat.json")
