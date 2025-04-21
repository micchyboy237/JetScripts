import os
from pathlib import Path
from typing import TypedDict, Literal, Annotated, List, Dict
from jet.logger import CustomLogger, logger
from jet.llm.ollama.base import initialize_ollama_settings
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.prebuilt import create_react_agent
from langchain.embeddings import init_embeddings
from langmem import create_multi_prompt_optimizer
from jet.llm.models import OLLAMA_MODEL_EMBEDDING_TOKENS


def initialize_task_environment(script_path: str, model_name: str = "ollama:llama3.1") -> tuple[CustomLogger, InMemoryStore, any]:
    """
    Initialize the environment, logger, and memory store for the task management agent.

    Args:
        script_path (str): Path to the script file for logging purposes.
        model_name (str): Name of the LLM model to initialize.

    Returns:
        tuple: Logger, InMemoryStore, and initialized LLM.
    """
    script_dir = os.path.dirname(os.path.abspath(script_path))
    log_file = os.path.join(
        script_dir, f"{os.path.splitext(os.path.basename(script_path))[0]}.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    initialize_ollama_settings()
    llm = init_chat_model("ollama:llama3.1")
    store = InMemoryStore(index={
        "dims": OLLAMA_MODEL_EMBEDDING_TOKENS["mxbai-embed-large"],
        "embed": init_embeddings("ollama:mxbai-embed-large")
    })

    return logger, store, llm


def create_task_tools(user_id_template: str = "{langgraph_user_id}") -> List:
    """
    Create tools for the task management agent, including task creation, assignment, and memory tools.

    Args:
        user_id_template (str): Template for user ID in memory tool namespaces.

    Returns:
        List: List of tools for the agent.
    """
    @tool
    def create_task(title: str, description: str, priority: Literal["low", "medium", "high"]) -> str:
        """Create a new task with a title, description, and priority."""
        logger.debug(
            f"Creating task: {title}\nDescription: {description}\nPriority: {priority}")
        return f"Task '{title}' created with priority {priority}"

    @tool
    def assign_task(task_title: str, assignee: str) -> str:
        """Assign a task to a team member."""
        logger.debug(f"Assigning task '{task_title}' to {assignee}")
        return f"Task '{task_title}' assigned to {assignee}"

    manage_memory_tool = create_manage_memory_tool(
        namespace=("task_assistant", user_id_template, "collection"))
    search_memory_tool = create_search_memory_tool(
        namespace=("task_assistant", user_id_template, "collection"))

    return [create_task, assign_task, manage_memory_tool, search_memory_tool]


def format_task_few_shot_examples(examples: List) -> str:
    """
    Format few-shot examples for the task prioritization prompt.

    Args:
        examples (List): List of example objects with task details and priority.

    Returns:
        str: Formatted string of examples.
    """
    formatted_examples = []
    for eg in examples:
        task = eg.value['task']
        priority = eg.value['priority']
        formatted_examples.append(
            f"Task: {task['title']}\nDescription: {task['description'][:200]}...\nPriority: {priority}"
        )
    return "\n\n".join(formatted_examples)


def prioritize_task(state: Dict, config: Dict, store: InMemoryStore, prompt_template: str) -> Dict:
    """
    Prioritize a task based on its details and examples in memory.

    Args:
        state (Dict): State containing task input.
        config (Dict): Configuration with user ID.
        store (InMemoryStore): Memory store for examples.
        prompt_template (str): Template for the prioritization prompt.

    Returns:
        Dict: Prioritization result with priority level.
    """
    class Prioritizer(BaseModel):
        reasoning: str = Field(
            description="Step-by-step reasoning behind the prioritization.")
        priority: Literal["low", "medium", "high"] = Field(
            description="The priority of the task: 'low', 'medium', or 'high'."
        )

    task = state["task_input"]
    user_id = config["configurable"]["langgraph_user_id"]
    namespace = ("task_assistant", user_id, "examples")
    examples = store.search(namespace, query=str(task))
    formatted_examples = format_task_few_shot_examples(examples)
    prompt = PromptTemplate.from_template(prompt_template).format(
        examples=formatted_examples, **task)
    messages = [HumanMessage(content=prompt)]
    llm_prioritizer = init_chat_model(
        "ollama:llama3.1").with_structured_output(Prioritizer)
    result = llm_prioritizer.invoke(messages)
    return {"priority_result": result.priority}


def create_task_agent(store: InMemoryStore, task_agent, prompt_template: str) -> any:
    """
    Create and compile the task management agent.

    Args:
        store (InMemoryStore): Memory store for prompts and examples.
        llm (any): Initialized language model.
        tools (List): List of tools for the agent.
        prompt_template (str): Initial system prompt for the task agent.

    Returns:
        any: Compiled LangGraph agent.
    """

    workflow = StateGraph(State)
    workflow.add_node("prioritize", lambda state, config: prioritize_task(
        state, config, store, prompt_template))
    workflow.add_node("task_agent", task_agent)

    def route_based_on_priority(state):
        return "task_agent" if state["priority_result"] in ["medium", "high"] else END

    workflow.add_edge(START, "prioritize")
    workflow.add_conditional_edges("prioritize", route_based_on_priority, {
                                   "task_agent": "task_agent", END: END})

    return workflow.compile(store=store)


def optimize_task_prompts(feedback: str, config: Dict, store: InMemoryStore, llm: any) -> str:
    """
    Optimize task prioritization and response prompts based on feedback.

    Args:
        feedback (str): Feedback to improve prompts.
        config (Dict): Configuration with user ID.
        store (InMemoryStore): Memory store for prompts.
        llm (any): Initialized language model.

    Returns:
        str: Status message indicating prompt optimization.
    """
    user_id = config["configurable"]["langgraph_user_id"]
    prioritize_prompt = store.get(
        ("task_assistant", user_id, "prompts"), "prioritize_prompt").value
    task_prompt = store.get(
        ("task_assistant", user_id, "prompts"), "task_prompt").value

    sample_task = {
        "title": "Update API documentation",
        "description": "Revise the API documentation to include missing endpoints and clarify usage examples.",
        "deadline": "2025-05-01"
    }

    optimizer = create_multi_prompt_optimizer(llm)
    conversation = [
        {"role": "system", "content": task_prompt},
        {"role": "user", "content": f"I have this task: {sample_task}"},
        {"role": "assistant", "content": "How should I prioritize or assign this task?"}
    ]
    prompts = [
        {"name": "prioritize", "prompt": prioritize_prompt},
        {"name": "task", "prompt": task_prompt}
    ]

    try:
        trajectories = [(conversation, {"feedback": feedback})]
        result = optimizer.invoke(
            {"trajectories": trajectories, "prompts": prompts})
        improved_prioritize_prompt = next(p["prompt"]
                                          for p in result if p["name"] == "prioritize")
        improved_task_prompt = next(p["prompt"]
                                    for p in result if p["name"] == "task")
    except Exception as e:
        logger.debug(f"API error: {e}")
        logger.debug("Using manual prompt improvement as fallback")
        improved_prioritize_prompt = prioritize_prompt + \
            "\n\nNote: Tasks related to API documentation or critical updates are high priority and should ALWAYS be classified as 'high'."
        improved_task_prompt = task_prompt + \
            "\n\nWhen handling tasks about documentation or critical updates, provide specific assignment suggestions and urgency notes."

    store.put(("task_assistant", user_id, "prompts"),
              "prioritize_prompt", improved_prioritize_prompt)
    store.put(("task_assistant", user_id, "prompts"),
              "task_prompt", improved_task_prompt)
    logger.debug(
        f"Prioritize prompt improved: {improved_prioritize_prompt[:100]}...")
    logger.debug(f"Task prompt improved: {improved_task_prompt[:100]}...")
    return "Prompts improved based on feedback!"


class State(TypedDict):
    task_input: dict
    messages: Annotated[list, add_messages]
    priority_result: str


def create_agent_prompt(state, config, store):
    messages = state['messages']
    user_id = config["configurable"]["langgraph_user_id"]
    system_prompt = store.get(
        ("task_assistant", user_id, "prompts"), "task_prompt").value
    return [{"role": "system", "content": system_prompt}] + messages


if __name__ == "__main__":
    # Example usage
    logger, store, llm = initialize_task_environment(__file__)
    tools = create_task_tools()
    prompt_template = """
    Prioritize the following task based on its title, description, and deadline.
    Use the provided examples to guide your decision.

    Examples:
    {examples}

    Task:
    Title: {title}
    Description: {description}
    Deadline: {deadline}

    Provide a priority level: low, medium, or high.
    """

    task_agent = create_react_agent(
        tools=tools, prompt=create_agent_prompt, store=store, model=llm)

    agent = create_task_agent(store, task_agent, prompt_template)
    config = {"configurable": {"langgraph_user_id": "user_123"}}

    # Example task input
    task_input = {
        "title": "Refactor authentication module",
        "description": "Update the authentication module to support OAuth 2.0 and improve security.",
        "deadline": "2025-06-01"
    }

    # Run the agent
    result = task_agent.invoke(
        {"task_input": task_input, "messages": []}, config)
    logger.info(f"Agent result: {result}")
