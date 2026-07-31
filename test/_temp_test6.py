import operator
from typing import Annotated, List, TypedDict

from jet.adapters.langchain.factory import get_chat_openai
from langchain_core.messages import HumanMessage

# from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent


# State with shared minimal context
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    research_plan: str  # Persisted high-level plan (external memory)
    sub_results: List[str]


llm = get_chat_openai(temperature=0)


# Sub-agent creator (isolated context)
def create_sub_agent(task: str):
    tools = [...]  # e.g., web search, calculator
    return create_react_agent(
        llm,
        tools,
        state_modifier=f"You are a specialized researcher for: {task}. Focus only on this.",
    )


# Lead/Orchestrator Node
def lead_agent(state: AgentState):
    # Plan and save to "memory" (external or state)
    plan_prompt = "Decompose the query into 2-4 parallel subtasks..."
    plan = llm.invoke(
        [HumanMessage(content=plan_prompt + str(state["messages"]))]
    ).content
    state["research_plan"] = plan

    # Spawn sub-agents (parallel in practice via concurrent execution)
    subtasks = extract_subtasks(plan)  # Custom parsing
    results = []
    for task in subtasks:
        sub_agent = create_sub_agent(task)
        sub_result = sub_agent.invoke({"messages": [HumanMessage(content=task)]})
        results.append(sub_result["messages"][-1].content)

    state["sub_results"] = results
    return state


# Synthesize Node
def synthesizer(state: AgentState):
    synthesis_prompt = f"Research plan: {state['research_plan']}\nSub-results: {state['sub_results']}\nSynthesize final answer."
    final = llm.invoke([HumanMessage(content=synthesis_prompt)]).content
    state["messages"].append(HumanMessage(content=final))
    return state


# Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("lead", lead_agent)
workflow.add_node("synthesize", synthesizer)
workflow.set_entry_point("lead")
workflow.add_edge("lead", "synthesize")
workflow.add_edge("synthesize", END)

graph = workflow.compile()
# Run: result = graph.invoke({"messages": [HumanMessage(content="Research AI agent token management strategies")]})
