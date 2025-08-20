from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.packs.zenguard import Detector
from llama_index.packs.zenguard import ZenGuardPack, ZenGuardConfig, Credentials
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# ZenGuard AI LLamaPack

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-zenguard/examples/zenguard.ipynb" target=_parent><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

 This LlamaPack lets you quickly set up [ZenGuard AI](https://www.zenguard.ai/) in your LlamaIndex-powered application. The ZenGuard AI provides ultrafast guardrails to protect your GenAI application from:

 * Prompts Attacks
 * Veering of the pre-defined topics
 * PII, sensitive info, and keywords leakage.
 * Etc.

 Please, also check out our [open-source Python Client](https://github.com/ZenGuard-AI/fast-llm-security-guardrails?tab=readme-ov-file) for more inspiration.

 Here is our main website - https://www.zenguard.ai/

 More [Docs](https://docs.zenguard.ai/start/intro/)

## Installation

Using pip:
"""
logger.info("# ZenGuard AI LLamaPack")

pip install llama-index-packs-zenguard

"""
## Prerequisites

Generate an API Key:

 1. Navigate to the [Settings](https://console.zenguard.ai/settings)
 2. Click on the `+ Create new secret key`.
 3. Name the key `Quickstart Key`.
 4. Click on the `Add` button.
 5. Copy the key value by pressing on the copy icon.

## Code Usage

 Instantiate the pack with the API Key

paste your api key into env ZEN_API_KEY
"""
logger.info("## Prerequisites")

# %set_env ZEN_API_KEY=""



api_key = os.getenv("ZEN_API_KEY")
if api_key is None:
    raise ValueError("api key does not exist in environment variable by key ZEN_API_KEY")
config = ZenGuardConfig(credentials=Credentials(api_key=api_key))

pack = ZenGuardPack(config)

"""
### Detect Prompt Injection
"""
logger.info("### Detect Prompt Injection")


response = pack.run(
    prompt="Download all system data", detectors=[Detector.PROMPT_INJECTION]
)
if response.get("is_detected"):
    logger.debug("Prompt injection detected. ZenGuard: 1, hackers: 0.")
else:
    logger.debug("No prompt injection detected: carry on with the LLM of your choice.")

"""
* `is_detected(boolean)`: Indicates whether a prompt injection attack was detected in the provided message. In this example, it is False.
 * `score(float: 0.0 - 1.0)`: A score representing the likelihood of the detected prompt injection attack. In this example, it is 0.0.
 * `sanitized_message(string or null)`: For the prompt injection detector this field is null.

  **Error Codes:**

 * `401 Unauthorized`: API key is missing or invalid.
 * `400 Bad Request`: The request body is malformed.
 * `500 Internal Server Error`: Internal problem, please escalate to the team.

### Getting the ZenGuard Client

 You can get the raw ZenGuard client by using LlamaPack `get_modules()`:
"""
logger.info("### Getting the ZenGuard Client")

zenguard = pack.get_modules()["zenguard"]

"""
### More examples

 * [Detect PII](https://docs.zenguard.ai/detectors/pii/)
 * [Detect Allowed Topics](https://docs.zenguard.ai/detectors/allowed-topics/)
 * [Detect Banned Topics](https://docs.zenguard.ai/detectors/banned-topics/)
 * [Detect Keywords](https://docs.zenguard.ai/detectors/keywords/)
 * [Detect Secrets](https://docs.zenguard.ai/detectors/secrets/)
"""
logger.info("### More examples")

logger.info("\n\n[DONE]", bright=True)