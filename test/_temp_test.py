import platform
from pathlib import Path
from jet.logger import CustomLogger
from jet.llm.ollama.base import initialize_ollama_settings
import os
from dotenv import load_dotenv
from typing import TypedDict, Literal, Annotated, List
from langgraph.graph import StateGraph, START, END, add_messages
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from pydantic import BaseModel, Field
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables.graph import MermaidDrawMethod
from langmem import create_multi_prompt_optimizer
from langchain.embeddings import init_embeddings
from jet.llm.models import OLLAMA_MODEL_EMBEDDING_TOKENS

# Initialize logging
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

# Initialize settings and environment
initialize_ollama_settings()
load_dotenv()

# Initialize LLM and store
llm = init_chat_model("ollama:llama3.1")
store = InMemoryStore(index={
    "dims": OLLAMA_MODEL_EMBEDDING_TOKENS["mxbai-embed-large"],
    "embed": init_embeddings("ollama:mxbai-embed-large")
})

# Define state and router


class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]
    triage_result: str


class Router(BaseModel):
    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification.")
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="The classification of an email: 'ignore', 'notify', or 'respond'."
    )


llm_router = llm.with_structured_output(Router)

# Reusable functions


def render_mermaid_graph(agent, output_filename="graph_output.png", draw_method=MermaidDrawMethod.API):
    """Render and save a Mermaid graph of the agent's workflow."""
    png_bytes = agent.get_graph().draw_mermaid_png(draw_method=draw_method)
    output_path = Path(output_filename)
    output_path.write_bytes(png_bytes)
    os.system(f"open {output_path}")


def format_few_shot_examples(examples):
    """Format few-shot examples for prompt construction."""
    formatted_examples = []
    for eg in examples:
        email = eg.value['email']
        label = eg.value['label']
        formatted_examples.append(
            f"From: {email['author']}\nSubject: {email['subject']}\nBody: {email['email_thread'][:300]}...\n\nClassification: {label}"
        )
    return "\n\n".join(formatted_examples)


def triage_email(state: State, config: dict, store: InMemoryStore) -> dict:
    """Triage an email based on its content and stored examples."""
    email = state["email_input"]
    user_id = config["configurable"]["langgraph_user_id"]
    namespace = ("email_assistant", user_id, "examples")
    examples = store.search(namespace, query=str(email))
    formatted_examples = format_few_shot_examples(examples)
    prompt = PromptTemplate.from_template(
        store.get(("email_assistant", user_id, "prompts"),
                  "triage_prompt").value
    ).format(examples=formatted_examples, **email)
    messages = [HumanMessage(content=prompt)]
    result = llm_router.invoke(messages)
    return {"triage_result": result.classification}


@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    logger.debug(
        f"Sending email to {to} with subject '{subject}'\nContent:\n{content}\n")
    return f"Email sent to {to} with subject '{subject}'"


@tool
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"


def create_agent_prompt(state, config, store):
    """Create a prompt for the response agent."""
    messages = state['messages']
    user_id = config["configurable"]["langgraph_user_id"]
    system_prompt = store.get(
        ("email_assistant", user_id, "prompts"), "response_prompt").value
    return [{"role": "system", "content": system_prompt}] + messages


def create_email_agent(store):
    """Create and compile an email processing agent."""
    tools = [
        write_email,
        check_calendar_availability,
        create_manage_memory_tool(namespace=(
            "email_assistant", "{langgraph_user_id}", "collection")),
        create_search_memory_tool(namespace=(
            "email_assistant", "{langgraph_user_id}", "collection"))
    ]
    response_agent = create_react_agent(
        tools=tools, prompt=create_agent_prompt, store=store, model=llm)
    workflow = StateGraph(State)
    workflow.add_node("triage", lambda state,
                      config: triage_email(state, config, store))
    workflow.add_node("response_agent", response_agent)
    workflow.add_edge(START, "triage")
    workflow.add_conditional_edges("triage", lambda state: "response_agent" if state["triage_result"] == "respond" else END,
                                   {"response_agent": "response_agent", END: END})
    return workflow.compile(store=store)


def optimize_prompts(feedback: str, config: dict, store: InMemoryStore):
    """Optimize triage and response prompts based on feedback."""
    user_id = config["configurable"]["langgraph_user_id"]
    triage_prompt = store.get(
        ("email_assistant", user_id, "prompts"), "triage_prompt").value
    response_prompt = store.get(
        ("email_assistant", user_id, "prompts"), "response_prompt").value
    sample_email = {
        "author": "Alice Smith <alice.smith@company.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Quick question about API documentation",
        "email_thread": "Hi John, I was reviewing the API documentation and noticed a few endpoints are missing. Could you help? Thanks, Alice",
    }
    optimizer = create_multi_prompt_optimizer(llm)
    conversation = [
        {"role": "system", "content": response_prompt},
        {"role": "user", "content": f"I received this email: {sample_email}"},
        {"role": "assistant", "content": "How can I assist you today?"}
    ]
    prompts = [
        {"name": "triage", "prompt": triage_prompt},
        {"name": "response", "prompt": response_prompt}
    ]
    try:
        trajectories = [(conversation, {"feedback": feedback})]
        result = optimizer.invoke(
            {"trajectories": trajectories, "prompts": prompts})
        improved_triage_prompt = next(p["prompt"]
                                      for p in result if p["name"] == "triage")
        improved_response_prompt = next(p["prompt"]
                                        for p in result if p["name"] == "response")
    except Exception as e:
        logger.debug(f"API error: {e}")
        logger.debug("Using manual prompt improvement as fallback")
        improved_triage_prompt = triage_prompt + \
            "\n\nNote: Emails about API documentation or missing endpoints are high priority and should ALWAYS be classified as 'respond'."
        improved_response_prompt = response_prompt + \
            "\n\nWhen responding to emails about documentation or API issues, acknowledge the specific issue mentioned and offer specific assistance rather than generic responses."
    store.put(("email_assistant", user_id, "prompts"),
              "triage_prompt", improved_triage_prompt)
    store.put(("email_assistant", user_id, "prompts"),
              "response_prompt", improved_response_prompt)
    logger.debug(f"Triage prompt improved: {improved_triage_prompt[:100]}...")
    logger.debug(
        f"Response prompt improved: {improved_response_prompt[:100]}...")
    return "Prompts improved based on feedback!"


def main():
    """Main function demonstrating usage of the email agent."""
    # Initialize prompts
    initial_triage_prompt = """Classify the email as 'ignore', 'respond', or 'notify' based on content."""
    initial_response_prompt = """You are a helpful assistant. Use the tools available, including memory tools, to assist the user."""
    store.put(("email_assistant", "test_user", "prompts"),
              "triage_prompt", initial_triage_prompt)
    store.put(("email_assistant", "test_user", "prompts"),
              "response_prompt", initial_response_prompt)

    # Create and render initial agent
    agent = create_email_agent(store)
    render_mermaid_graph(agent)

    # Process email before optimization
    email_input = {
        "author": "Alice Smith <alice.smith@company.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Quick question about API documentation",
    }
    config = {"configurable": {"langgraph_user_id": "test_user"}}
    inputs = {"email_input": email_input, "messages": []}
    logger.debug("\n\nProcessing original email BEFORE optimization...\n\n")
    for output in agent.stream(inputs, config=config):
        for key, value in output.items():
            logger.debug(f"-----\n{key}:")
            logger.debug(value)
        logger.debug("-----")

    # Add example to memory
    example1 = {
        "email": {
            "author": "Spammy Marketer <spam@example.com>",
            "to": "John Doe <john.doe@company.com>",
            "subject": "BIG SALE!!!",
            "email_thread": "Buy our product now and get 50% off!",
        },
        "label": "ignore",
    }
    store.put(("email_assistant", "test_user", "examples"),
              "spam_example", example1)

    # Add API documentation example
    api_doc_example = {
        "email": {
            "author": "Developer <dev@company.com>",
            "to": "John Doe <john.doe@company.com>",
            "subject": "API Documentation Issue",
            "email_thread": "Found missing endpoints in the API docs. Need urgent update.",
        },
        "label": "respond",
    }
    store.put(("email_assistant", "test_user", "examples"),
              "api_doc_example", api_doc_example)
    logger.debug("Added API documentation example to episodic memory")

    # Optimize prompts
    feedback = "Emails about API documentation should always be classified as 'respond' and receive specific responses."
    optimize_prompts(feedback, config, store)

    # Process email after optimization
    logger.debug(
        "\n\nProcessing the SAME email AFTER optimization with a fresh agent...\n\n")
    new_agent = create_email_agent(store)
    for output in new_agent.stream(inputs, config=config):
        for key, value in output.items():
            logger.debug(f"-----\n{key}:")
            logger.debug(value)
        logger.debug("-----")
    logger.info("\n\n[DONE]", bright=True)


if __name__ == "__main__":
    if platform.system() == "Emscripten":
        import asyncio
        asyncio.ensure_future(main())
    else:
        main()
