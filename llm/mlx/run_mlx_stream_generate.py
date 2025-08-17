import os
import shutil
from jet.llm.mlx.base import MLX
from jet.models.model_types import LLMModelType
from jet.file.utils import save_file
from jet.transformers.formatters import format_json
from jet.logger import logger


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), "generated", os.path.splitext(os.path.basename(__file__))[0])

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

MLX_LOG_DIR = f"{OUTPUT_DIR}/mlx-logs"

model: LLMModelType = "llama-3.2-1b-instruct-4bit"
seed = 42


"""Example of using the .stream_generate method for streaming chat completions."""
prompt = "Tell me a very short story about a brave knight."

logger.debug("Streaming Chat Response:")
response = ""

client = MLX(seed=seed)

for stream_response in client.stream_generate(
    prompt=prompt,
    model=model,
    # max_tokens=200,
    temperature=0.3,
    log_dir=MLX_LOG_DIR,
    verbose=True
):
    content = stream_response["choices"][0]["text"]
    response += content

    if stream_response["choices"][0]["finish_reason"]:
        logger.newline()


save_file({
    "prompt": prompt,
    "response": response
}, f"{OUTPUT_DIR}/stream_generate.json")
