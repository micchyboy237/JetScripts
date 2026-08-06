import os
from typing import Annotated, TypedDict

from jet.adapters.langchain.factory import get_chat_openai
from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.mem0.factory import get_memory_config
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from mem0 import Memory
from phoenix.otel import register

# ─── 1. PHOENIX TRACING SETUP ────────────────────────────────────────
# Points to YOUR running Phoenix gRPC endpoint from startup logs
tracer_provider = register(
    project_name="mem0-langgraph-dual-scope",
    endpoint=os.getenv(
        "LLM_OBS_PHOENIX_OTLP_GRPC_URL"
    ),  # gRPC endpoint from your Phoenix output
)

# ─── 2. SELF-HOSTED MEM0 CONFIG (NO CLOUD DEPENDENCY) ───────────────
MEM0_CONFIG = get_memory_config("mem0_langraph_memories_v1")

memory = Memory.from_config(MEM0_CONFIG)


# ─── 3. LANGGRAPH STATE DEFINITION ──────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    agent_id: str
    shared_context: str
    agent_context: str


# ─── 4. MEMORY NODES (AUTO-TRACED BY PHOENIX) ───────────────────────
def retrieve_shared_memory(state: AgentState) -> dict:
    """Retrieve cross-agent user memories. Appears as distinct span in Phoenix."""
    results = memory.search(
        state["messages"][-1].content if state["messages"] else "",
        user_id=state["user_id"],
    )
    facts = "\n".join(f"- {m['memory']}" for m in results.get("results", []))
    return {"shared_context": facts or "(no shared memories)"}


def retrieve_agent_memory(state: AgentState) -> dict:
    """Retrieve agent-isolated memories. Scoped by agent_id."""
    results = memory.search(
        state["messages"][-1].content if state["messages"] else "",
        user_id=state["user_id"],
        agent_id=state["agent_id"],
    )
    facts = "\n".join(f"- {m['memory']}" for m in results.get("results", []))
    return {"agent_context": facts or "(no agent-specific memories)"}


def generate_response(state: AgentState) -> dict:
    """LLM call with merged dual-scope context injection."""
    llm = get_chat_openai(model=LLM_MODEL, temperature=0)

    system_prompt = f"""You are the {state["agent_id"]} agent.

## Universal User Context (shared across ALL agents)
{state["shared_context"]}

## Your Specialized Memory (private to YOU only)
{state["agent_context"]}

Respond helpfully using both contexts. Never leak agent-specific details unless asked."""

    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            *state["messages"],
        ]
    )
    return {"messages": [response]}


def update_memory(state: AgentState) -> dict:
    """Persist new turn. Auto-classifies scope based on content heuristics."""
    last_two = state["messages"][-2:]

    # Simple heuristic: if message contains agent-specific keywords, isolate it
    agent_keywords = {"research", "paper", "code", "debug", "implementation"}
    content = last_two[-1].content.lower() if last_two else ""
    is_agent_specific = any(kw in content for kw in agent_keywords)

    if is_agent_specific:
        memory.add(last_two, user_id=state["user_id"], agent_id=state["agent_id"])
    else:
        memory.add(last_two, user_id=state["user_id"])

    return {}


# ─── 5. BUILD GRAPH ─────────────────────────────────────────────────
graph = StateGraph(AgentState)
graph.add_node("retrieve_shared", retrieve_shared_memory)
graph.add_node("retrieve_agent", retrieve_agent_memory)
graph.add_node("generate", generate_response)
graph.add_node("update_memory", update_memory)

graph.set_entry_point("retrieve_shared")
graph.add_edge("retrieve_shared", "retrieve_agent")
graph.add_edge("retrieve_agent", "generate")
graph.add_edge("generate", "update_memory")
graph.add_edge("update_memory", END)

app = graph.compile()

# ─── 6. RUN WITH DUAL-SCOPE DEMO ────────────────────────────────────
if __name__ == "__main__":
    print(f"🔍 Open {os.getenv('LLM_OBS_PHOENIX_URL')} to view traces in Phoenix")
    print("=" * 60)

    # Turn 1: Shared preference (all agents will see this)
    result = app.invoke(
        {
            "messages": [{"role": "user", "content": "I prefer TypeScript examples"}],
            "user_id": "alex",
            "agent_id": "coder",
            "shared_context": "",
            "agent_context": "",
        }
    )
    print(f"Coder: {result['messages'][-1].content}\n")

    # Turn 2: Agent-specific knowledge (only coder sees this)
    result = app.invoke(
        {
            "messages": [
                {"role": "user", "content": "Debug the React hook in auth.ts"}
            ],
            "user_id": "alex",
            "agent_id": "coder",
            "shared_context": "",
            "agent_context": "",
        }
    )
    print(f"Coder: {result['messages'][-1].content}\n")

    # Turn 3: Different agent accessing shared memory but NOT coder's debug context
    result = app.invoke(
        {
            "messages": [
                {"role": "user", "content": "What language should I use for docs?"}
            ],
            "user_id": "alex",
            "agent_id": "writer",
            "shared_context": "",
            "agent_context": "",
        }
    )
    print(f"Writer: {result['messages'][-1].content}")
    print(
        "\n✅ Check Phoenix UI → 'mem0-langgraph-dual-scope' project for full trace hierarchy"
    )
