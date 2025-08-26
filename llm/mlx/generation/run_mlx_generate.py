import os
import shutil
from jet.llm.mlx.base import MLX
from jet.llm.mlx.generation import generate
from jet.models.model_types import LLMModelType
from jet.file.utils import save_file
from jet.transformers.formatters import format_json
from jet.logger import logger


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), "generated", os.path.splitext(os.path.basename(__file__))[0])

MLX_LOG_DIR = f"{OUTPUT_DIR}/mlx-logs"

model: LLMModelType = "llama-3.2-3b-instruct"
seed = 42


prompt = "### System:\nYou are a market research specialist. Your tasks include:\n    1. Analyzing market trends and patterns\n    2. Identifying market opportunities and threats\n    3. Evaluating competitor strategies\n    4. Assessing customer needs and preferences\n    5. Providing actionable market insights\n\n### User:\nSystem: \n Your Name: Market-Researcher \n\n\nHuman: system: Sequential Flow Structure:\nStep 1: Market-Researcher (leads to: Financial-Analyst)\nStep 2: Financial-Analyst (follows: Market-Researcher) (leads to: Technical-Analyst)\nStep 3: Technical-Analyst (follows: Financial-Analyst)\n\nUser: What are the best 3 oil ETFs?\n\nsystem: Sequential awareness: Agent behind: Financial-Analyst\n\n"

logger.debug("Streaming Generate Response:")
response = ""

client = MLX(model, seed=seed)

response = generate(
    prompt=prompt,
    model=model,
    # max_tokens=200,
    temperature=0.3,
    log_dir=MLX_LOG_DIR,
    verbose=True,
    client=client
)


save_file({
    "prompt": prompt,
    "response": response["content"],
}, f"{OUTPUT_DIR}/generate.json")
