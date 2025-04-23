import os
from jet.llm.models import OLLAMA_MODEL_EMBEDDING_TOKENS
from jet.logger import CustomLogger
from jet.llm.ollama.base import initialize_ollama_settings
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from langmem import create_manage_memory_tool, create_search_memory_tool
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.config import get_store


script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

initialize_ollama_settings()

logger.orange("""# Semantic Memory""")


checkpointer = InMemorySaver()
store = InMemoryStore()

agent = create_react_agent("ollama:llama3.1", tools=[],
                           store=store, checkpointer=checkpointer)


def chat(agent, txt, thread_id):
    result_state = agent.invoke({"messages": [{"role": "user", "content": txt}]}, config={
                                "configurable": {"thread_id": thread_id}})
    return result_state["messages"][-1].content


thread_1 = "thread-1"
chat(agent, "Hi there, I'm training for a half marathon in 2 months - could you propose a daily training plan to prepare?", thread_1)

chat(agent, "Nice! Wish me luck!", thread_1)

thread_2 = "thread-2"
chat(agent, "Nice! Oh thank you! It'll be hard.", thread_2)

logger.orange("""## Adding Memory

We'll use regular tools.""")


store = InMemoryStore(
    index={
        "dims": OLLAMA_MODEL_EMBEDDING_TOKENS["mxbai-embed-large"],
        "embed": "ollama:mxbai-embed-large"
    }
)

namespace = ("agent_memories",)
memory_tools = [
    create_manage_memory_tool(namespace),
    create_search_memory_tool(namespace)
]
checkpointer = InMemorySaver()
agent = create_react_agent(
    "ollama:llama3.1", tools=memory_tools, store=store, checkpointer=checkpointer)

thread_1 = "thread-1"
chat(agent, "Hi there, I'm training for a half marathon in 2 months - could you propose a daily training plan to prepare?", thread_1)

logger.debug(chat(
    agent, "Nice! Wish me luck! Please note down the detailed memories for me :)", thread_1))

thread_2 = "thread-2"
chat(agent, "Remember what I'm supposed to do for my training this week? It's week 3...", thread_2)

chat(agent, "That may be tricky. I just sprained my ankle. Could you update my plan to include more cross training? Be sure to update the existing key of our plan", thread_2)

logger.orange("""## Different User Memories""")


store = InMemoryStore(
    index={
        "dims": OLLAMA_MODEL_EMBEDDING_TOKENS["mxbai-embed-large"],
        "embed": "ollama:mxbai-embed-large"
    }
)

namespace = ("agent_memories", "{user_id}")
memory_tools = [
    create_manage_memory_tool(namespace),
    create_search_memory_tool(namespace)
]
checkpointer = InMemorySaver()

agent = create_react_agent(
    "ollama:llama3.1", tools=memory_tools, store=store, checkpointer=checkpointer)


def chat(agent, txt, thread_id, user_id):
    result_state = agent.invoke({"messages": [{"role": "user", "content": txt}]},
                                config={"configurable": {"thread_id": thread_id, "user_id": user_id}})
    return result_state["messages"][-1].content


thread_1 = "thread-1"
user_id = "User-A"
chat(agent,
     "Hi I'm Will, I'm training for a half marathon in 2 months - could you propose a daily training plan to prepare and help me stay honest??",
     thread_1,
     user_id)

thread_1 = "thread-2"
user_id2 = "User-B"
chat(agent,
     "Hi I'm John, I'm learning chess - could you help me become great??",
     thread_1,
     user_id2)

chat(agent,
     "Do you remember me liking any sports?",
     thread_1,
     user_id2)

items = store.search(("agent_memories",))
for item in items:
    logger.debug(item.namespace, item.value)

logger.orange("""## "Eager" memory retrieval

We can fetch memories before the first LLM call to simplify its response. Otherwise, it has known and unknown unknowns so will almost always try to search for some subclass of questions.""")


store = InMemoryStore(
    index={
        "dims": OLLAMA_MODEL_EMBEDDING_TOKENS["mxbai-embed-large"],
        "embed": "ollama:mxbai-embed-large"
    }
)

namespace = ("agent_memories",)
memory_tools = [
    create_manage_memory_tool(namespace),
    create_search_memory_tool(namespace)
]
checkpointer = InMemorySaver()


def prompt(state):
    store = get_store()
    items = store.search(namespace, query=state["messages"][-1].content)
    memories = "\n\n".join(str(item) for item in items)
    system_msg = {"role": "system", "content": f"## Memories:\n\n{memories}"}
    return [system_msg] + state["messages"]


agent = create_react_agent("ollama:llama3.1", prompt=prompt,
                           tools=memory_tools, store=store, checkpointer=checkpointer)

thread_1 = "thread-1"
chat(agent, "Hi there, I'm training for a half marathon in 2 months - could you propose a daily training plan to prepare?", thread_1, None)

logger.debug(chat(
    agent, "Nice! Wish me luck! Please note down the detailed memories for me :)", thread_1, None))

thread_2 = "thread-2"
chat(agent, "What I'm supposed to do for my training this week? It's week 3...", thread_2, None)

logger.info("\n\n[DONE]", bright=True)
