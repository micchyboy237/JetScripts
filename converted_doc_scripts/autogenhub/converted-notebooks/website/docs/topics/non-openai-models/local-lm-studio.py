from autogen import ConversableAgent
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# LM Studio

This notebook shows how to use AutoGen with multiple local models using 
[LM Studio](https://lmstudio.ai/)'s multi-model serving feature, which is available since
version 0.2.17 of LM Studio.

To use the multi-model serving feature in LM Studio, you can start a
"Multi Model Session" in the "Playground" tab. Then you select relevant
models to load. Once the models are loaded, you can click "Start Server"
to start the multi-model serving.
The models will be available at a locally hosted Ollama-compatible endpoint.

## Two Agent Chats

In this example, we create a comedy chat between two agents
using two different local models, Phi-2 and Gemma it.

We first create configurations for the models.
"""
logger.info("# LM Studio")

gemma = {
    "config_list": [
        {
            "model": "lmstudio-ai/gemma-2b-it-GGUF/gemma-2b-it-q8_0.gguf:0",
            "base_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
        },
    ],
    "cache_seed": None,  # Disable caching.
}

phi2 = {
    "config_list": [
        {
            "model": "TheBloke/phi-2-GGUF/phi-2.Q4_K_S.gguf:0",
            "base_url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
        },
    ],
    "cache_seed": None,  # Disable caching.
}

"""
Now we create two agents, one for each model.
"""
logger.info("Now we create two agents, one for each model.")


jack = ConversableAgent(
    "Jack (Phi-2)",
    llm_config=phi2,
    system_message="Your name is Jack and you are a comedian in a two-person comedy show.",
)
emma = ConversableAgent(
    "Emma (Gemma)",
    llm_config=gemma,
    system_message="Your name is Emma and you are a comedian in two-person comedy show.",
)

"""
Now we start the conversation.
"""
logger.info("Now we start the conversation.")

chat_result = jack.initiate_chat(emma, message="Emma, tell me a joke.", max_turns=2)

logger.info("\n\n[DONE]", bright=True)