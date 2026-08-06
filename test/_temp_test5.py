import os
import time
import warnings
from typing import Annotated, TypedDict

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")

import logging

from jet.adapters.langchain.factory import get_chat_openai
from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.mem0.factory import get_memory_config
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from mem0 import Memory
from opentelemetry import trace
from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
from opentelemetry.trace import Status, StatusCode
from phoenix.otel import register
from rich.console import Console
from rich.logging import RichHandler

# ---------------------------------------------------------------------------
# Observability Setup (mirrors chat_stream_observability pattern)
# ---------------------------------------------------------------------------
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, markup=True, rich_tracebacks=True)],
)
logger = logging.getLogger("mem0-langgraph-dual-scope")

PHOENIX_URL = os.getenv("LLM_OBS_PHOENIX_URL", "http://localhost:6006")


def setup_observability(project_name: str = "mem0-langgraph-dual-scope"):
    """Configure OpenTelemetry following chat_stream_observability pattern."""
    # Critical: Enable content capture for prompts/responses in Phoenix
    os.environ.setdefault(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_AND_EVENT"
    )

    tracer_provider = register(
        project_name=project_name,
        endpoint=f"{PHOENIX_URL}/v1/traces",
        batch=False,
    )

    # Auto-instrument OpenAI/LangChain OpenAI calls for token/model metadata
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

    console.print(f"🔭 Observability → [link={PHOENIX_URL}]{PHOENIX_URL}[/link]")
    logger.info(f"📁 Phoenix project: {project_name}")
    return tracer_provider


tracer_provider = setup_observability()
tracer = trace.get_tracer(__name__)

# ---------------------------------------------------------------------------
# Mem0 + LangGraph Setup
# ---------------------------------------------------------------------------
MEM0_CONFIG = get_memory_config("mem0_langraph_memories_v1")
memory = Memory.from_config(MEM0_CONFIG)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    agent_id: str
    shared_context: str
    agent_context: str


def _serialize_messages(messages: list[BaseMessage]) -> list[dict]:
    """Convert LangChain messages to plain dicts for Mem0 compatibility.
    Mem0's parse_vision_messages() calls msg.get("role"), which fails
    on Pydantic-backed LangChain message objects (HumanMessage, AIMessage).
    """
    role_map = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "tool": "tool",
    }
    serialized = []
    for msg in messages:
        if isinstance(msg, BaseMessage):
            serialized.append(
                {
                    "role": role_map.get(msg.type, msg.type),
                    "content": msg.content
                    if isinstance(msg.content, str)
                    else str(msg.content),
                }
            )
        elif isinstance(msg, dict):
            serialized.append(msg)
    return serialized


# ---------------------------------------------------------------------------
# Graph Nodes (each wrapped with observability spans)
# ---------------------------------------------------------------------------
def retrieve_shared_memory(state: AgentState) -> dict:
    """Retrieve cross-agent user memories with observability span."""
    with tracer.start_as_current_span(
        "memory.retrieve_shared",
        attributes={"user_id": state["user_id"], "memory.scope": "shared"},
    ) as span:
        query = state["messages"][-1].content if state["messages"] else ""
        span.set_attribute("memory.query", query[:500])

        t0 = time.perf_counter()
        try:
            results = memory.search(query=query, filters={"user_id": state["user_id"]})
            duration = time.perf_counter() - t0

            facts = "\n".join(f"- {m['memory']}" for m in results.get("results", []))
            span.set_attribute("memory.result_count", len(results.get("results", [])))
            span.set_attribute("memory.duration_s", round(duration, 4))
            span.set_status(Status(StatusCode.OK))

            logger.info(
                f"🧠 Shared memory: {len(results.get('results', []))} facts ({duration:.3f}s)"
            )
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            logger.exception("❌ Shared memory retrieval failed")
            raise

        return {"shared_context": facts or "(no shared memories)"}


def retrieve_agent_memory(state: AgentState) -> dict:
    """Retrieve agent-isolated memories with observability span."""
    with tracer.start_as_current_span(
        "memory.retrieve_agent",
        attributes={
            "user_id": state["user_id"],
            "agent_id": state["agent_id"],
            "memory.scope": "agent",
        },
    ) as span:
        query = state["messages"][-1].content if state["messages"] else ""
        span.set_attribute("memory.query", query[:500])

        t0 = time.perf_counter()
        try:
            results = memory.search(
                query=query,
                filters={"user_id": state["user_id"], "agent_id": state["agent_id"]},
            )
            duration = time.perf_counter() - t0

            facts = "\n".join(f"- {m['memory']}" for m in results.get("results", []))
            span.set_attribute("memory.result_count", len(results.get("results", [])))
            span.set_attribute("memory.duration_s", round(duration, 4))
            span.set_status(Status(StatusCode.OK))

            logger.info(
                f"🤖 Agent memory [{state['agent_id']}]: "
                f"{len(results.get('results', []))} facts ({duration:.3f}s)"
            )
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            logger.exception("❌ Agent memory retrieval failed")
            raise

        return {"agent_context": facts or "(no agent-specific memories)"}


def generate_response(state: AgentState) -> dict:
    """LLM call — auto-instrumented by OpenAIInstrumentor + parent span."""
    with tracer.start_as_current_span(
        "agent.generate",
        attributes={"agent_id": state["agent_id"], "user_id": state["user_id"]},
    ) as span:
        llm = get_chat_openai(model=LLM_MODEL, temperature=0)
        system_prompt = f"""You are the {state["agent_id"]} agent.
## Universal User Context (shared across ALL agents)
{state["shared_context"]}
## Your Specialized Memory (private to YOU only)
{state["agent_context"]}
Respond helpfully using both contexts. Never leak agent-specific details unless asked."""

        try:
            response = llm.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    *state["messages"],
                ]
            )
            span.set_attribute("llm.response_length", len(str(response.content)))
            span.set_status(Status(StatusCode.OK))
            logger.info(f"💬 Generated response ({len(str(response.content))} chars)")
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            logger.exception("❌ LLM generation failed")
            raise

        return {"messages": [response]}


def update_memory(state: AgentState) -> dict:
    """Persist new turn with observability span."""
    with tracer.start_as_current_span(
        "memory.update",
        attributes={"user_id": state["user_id"], "agent_id": state["agent_id"]},
    ) as span:
        last_two = state["messages"][-2:]
        if not last_two:
            span.set_attribute("memory.skipped", True)
            span.set_status(Status(StatusCode.OK))
            return {}

        serialized_messages = _serialize_messages(last_two)
        content = (
            last_two[-1].content.lower()
            if isinstance(last_two[-1].content, str)
            else ""
        )
        agent_keywords = {"research", "paper", "code", "debug", "implementation"}
        is_agent_specific = any(kw in content for kw in agent_keywords)

        t0 = time.perf_counter()
        try:
            if is_agent_specific:
                memory.add(
                    serialized_messages,
                    user_id=state["user_id"],
                    agent_id=state["agent_id"],
                )
                scope = "agent"
            else:
                memory.add(serialized_messages, user_id=state["user_id"])
                scope = "shared"

            duration = time.perf_counter() - t0
            span.set_attribute("memory.scope_written", scope)
            span.set_attribute("memory.duration_s", round(duration, 4))
            span.set_status(Status(StatusCode.OK))

            logger.info(f"💾 Memory saved ({scope}) in {duration:.3f}s")
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            logger.exception("❌ Memory update failed")
            raise

        return {}


# ---------------------------------------------------------------------------
# Graph Compilation
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    console.print(f"🔍 Open [link={PHOENIX_URL}]{PHOENIX_URL}[/link] to view traces")
    console.print("=" * 60)

    test_cases = [
        {
            "messages": [{"role": "user", "content": "I prefer TypeScript examples"}],
            "user_id": "alex",
            "agent_id": "coder",
        },
        {
            "messages": [
                {"role": "user", "content": "Debug the React hook in auth.ts"}
            ],
            "user_id": "alex",
            "agent_id": "coder",
        },
        {
            "messages": [
                {"role": "user", "content": "What language should I use for docs?"}
            ],
            "user_id": "alex",
            "agent_id": "writer",
        },
    ]

    for i, tc in enumerate(test_cases, 1):
        with tracer.start_as_current_span(
            f"test_case_{i}", attributes={"agent_id": tc["agent_id"]}
        ) as span:
            trace_id = span.get_span_context().trace_id
            trace_url = (
                f"{PHOENIX_URL.rstrip('/')}/redirects/traces/{format(trace_id, '032x')}"
            )
            console.print(f"\n🔗 Test {i} trace: [link={trace_url}]{trace_url}[/link]")

            result = app.invoke({**tc, "shared_context": "", "agent_context": ""})
            agent_label = tc["agent_id"].capitalize()
            console.print(
                f"[bold green]{agent_label}:[/bold green] "
                f"{result['messages'][-1].content}\n"
            )

            span.set_status(Status(StatusCode.OK))

    console.print(
        "\n✅ Check Phoenix UI → '[bold]mem0-langgraph-dual-scope[/bold]' project"
    )
