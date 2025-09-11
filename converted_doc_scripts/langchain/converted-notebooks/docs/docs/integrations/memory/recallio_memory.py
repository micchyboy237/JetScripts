from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.prompts import ChatPromptTemplate
from langchain_recallio.memory import RecallioMemory
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
# RecallioMemory + LangChain Integration Demo
A minimal notebook to show drop-in usage of RecallioMemory in LangChain (with scoped writes and recall).
"""
logger.info("# RecallioMemory + LangChain Integration Demo")

# %pip install recallio langchain langchain-recallio ollama

"""
## Setup: API Keys & Imports
"""
logger.info("## Setup: API Keys & Imports")


RECALLIO_API_KEY = os.getenv("RECALLIO_API_KEY", "YOUR_RECALLIO_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

"""
## Initialize RecallioMemory
"""
logger.info("## Initialize RecallioMemory")

memory = RecallioMemory(
    project_id="project_abc",
    api_key=RECALLIO_API_KEY,
    session_id="demo-session-001",
    user_id="demo-user-42",
    default_tags=["test", "langchain"],
    return_messages=True,
)

"""
## Build a LangChain ConversationChain with RecallioMemory
"""
logger.info("## Build a LangChain ConversationChain with RecallioMemory")

# llm = ChatOllama(model="llama3.2")
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "The following is a friendly conversation between a human and an AI. "
            "The AI is talkative and provides lots of specific details from its context. "
            "If the AI does not know the answer to a question, it truthfully says it does not know.",
        ),
        ("placeholder", "{history}"),  # RecallioMemory will fill this slot
        ("human", "{input}"),
    ]
)

base_chain = prompt | llm


def chat_with_memory(user_input: str):
    memory_vars = memory.load_memory_variables({"input": user_input})

    response = base_chain.invoke(
        {"input": user_input, "history": memory_vars.get("history", "")}
    )

    memory.save_context({"input": user_input}, {"output": response.content})

    return response

"""
## Example: Chat with Memory
"""
logger.info("## Example: Chat with Memory")

resp1 = chat_with_memory("Hi! My name is Guillaume. Remember that.")
logger.debug("Bot:", resp1.content)

resp2 = chat_with_memory("What is my name?")
logger.debug("Bot:", resp2.content)

"""
## See What Is Stored in Recallio
This is for debugging/demo only; in production, you wouldn't do this on every run.
"""
logger.info("## See What Is Stored in Recallio")

logger.debug("Current memory variables:", memory.load_memory_variables({}))

"""
## Clear Memory (Optional Cleanup - Requires Manager level Key)
"""
logger.info("## Clear Memory (Optional Cleanup - Requires Manager level Key)")


logger.info("\n\n[DONE]", bright=True)