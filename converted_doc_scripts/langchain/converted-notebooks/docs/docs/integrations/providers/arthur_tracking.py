from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_community.callbacks import ArthurCallbackHandler
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.messages import HumanMessage
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
# Arthur

>[Arthur](https://arthur.ai) is a model monitoring and observability platform.

The following guide shows how to run a registered chat LLM with the Arthur callback handler to automatically log model inferences to Arthur.

If you do not have a model currently onboarded to Arthur, visit our [onboarding guide for generative text models](https://docs.arthur.ai/user-guide/walkthroughs/model-onboarding/generative_text_onboarding.html). For more information about how to use the `Arthur SDK`, visit our [docs](https://docs.arthur.ai/).

## Installation and Setup

Place Arthur credentials here
"""
logger.info("# Arthur")

arthur_url = "https://app.arthur.ai"
arthur_login = "your-arthur-login-username-here"
arthur_model_id = "your-arthur-model-id-here"

"""
## Callback handler
"""
logger.info("## Callback handler")


"""
Create Langchain LLM with Arthur callback handler
"""
logger.info("Create Langchain LLM with Arthur callback handler")

def make_langchain_chat_llm():
    return ChatOllama(
        streaming=True,
        temperature=0.1,
        callbacks=[
            StreamingStdOutCallbackHandler(),
            ArthurCallbackHandler.from_credentials(
                arthur_model_id, arthur_url=arthur_url, arthur_login=arthur_login
            ),
        ],
    )

chatgpt = make_langchain_chat_llm()

"""
Running the chat LLM with this `run` function will save the chat history in an ongoing list so that the conversation can reference earlier messages and log each response to the Arthur platform. You can view the history of this model's inferences on your [model dashboard page](https://app.arthur.ai/).

Enter `q` to quit the run loop
"""
logger.info("Running the chat LLM with this `run` function will save the chat history in an ongoing list so that the conversation can reference earlier messages and log each response to the Arthur platform. You can view the history of this model's inferences on your [model dashboard page](https://app.arthur.ai/).")

def run(llm):
    history = []
    while True:
        user_input = input("\n>>> input >>>\n>>>: ")
        if user_input == "q":
            break
        history.append(HumanMessage(content=user_input))
        history.append(llm(history))

run(chatgpt)

logger.info("\n\n[DONE]", bright=True)