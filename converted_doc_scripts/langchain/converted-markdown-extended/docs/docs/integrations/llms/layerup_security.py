from jet.adapters.langchain.chat_ollama import Ollama
from jet.logger import logger
from langchain_community.llms.layerup_security import LayerupSecurity
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# Layerup Security

The [Layerup Security](https://uselayerup.com) integration allows you to secure your calls to any LangChain LLM, LLM chain or LLM agent. The LLM object wraps around any existing LLM object, allowing for a secure layer between your users and your LLMs.

While the Layerup Security object is designed as an LLM, it is not actually an LLM itself, it simply wraps around an LLM, allowing it to adapt the same functionality as the underlying LLM.

## Setup
First, you'll need a Layerup Security account from the Layerup [website](https://uselayerup.com).

Next, create a project via the [dashboard](https://dashboard.uselayerup.com), and copy your API key. We recommend putting your API key in your project's environment.

Install the Layerup Security SDK:
"""
logger.info("# Layerup Security")

pip install LayerupSecurity

"""
And install LangChain Community:
"""
logger.info("And install LangChain Community:")

pip install langchain-community

"""
And now you're ready to start protecting your LLM calls with Layerup Security!
"""
logger.info("And now you're ready to start protecting your LLM calls with Layerup Security!")


ollama = Ollama(
    model_name="gpt-3.5-turbo",
#     ollama_)

layerup_security = LayerupSecurity(
    llm=ollama,

    layerup_layerup_api_base_url="https://api.uselayerup.com/v1",

    prompt_guardrails=[],

    response_guardrails=["layerup.hallucination"],

    mask=False,

    metadata={"customer": "example@uselayerup.com"},

    handle_prompt_guardrail_violation=(
        lambda violation: {
            "role": "assistant",
            "content": (
                "There was sensitive data! I cannot respond. "
                "Here's a dynamic canned response. Current date: {}"
            ).format(datetime.now())
        }
        if violation["offending_guardrail"] == "layerup.sensitive_data"
        else None
    ),

    handle_response_guardrail_violation=(
        lambda violation: {
            "role": "assistant",
            "content": (
                "Custom canned response with dynamic data! "
                "The violation rule was {}."
            ).format(violation["offending_guardrail"])
        }
    ),
)

response = layerup_security.invoke(
    "Summarize this message: my name is Bob Dylan. My SSN is 123-45-6789."
)

logger.info("\n\n[DONE]", bright=True)