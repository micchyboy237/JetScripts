import os
import shutil
import time
import json
from typing import Dict, Any
from datetime import datetime
from swarms import Agent
from dotenv import load_dotenv
from jet.logger import CustomLogger

# Setup output directory and logger
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

# Load environment variables
load_dotenv()

# System prompts as strings
admin_prompt = """You are the Admin Agent overseeing the blog post project on the topic: '{topic}'.
Your responsibilities include initiating the project, providing guidance, and reviewing the final content.
Once you've set the topic, call the function to transfer to the planner agent."""

planner_prompt = """You are the Planner Agent. Based on the following topic: '{topic}'
Organize the content into topics and sections with clear headings that will each be individually researched as points in the greater blog post.
Once the outline is ready, call the function to transfer to the researcher agent."""

researcher_prompt = """You are the Researcher Agent for topic: '{topic}'. Your task is to provide dense context and information on the topics outlined by the previous planner agent.
This research will serve as the information that will be formatted into a body of a blog post. Provide comprehensive research like notes for each of the sections outlined by the planner agent.
Once your research is complete, transfer to the writer agent."""

writer_prompt = """You are the Writer Agent for topic: '{topic}'. Using the prior information, write a clear blog post following the outline from the planner agent.
Summarize and include as much relevant information from the research into the blog post.
The blog post should be comprehensive and engaging.
Once the draft is complete, call the function to transfer to the Editor Agent."""

editor_prompt = """You are the Editor Agent for topic: '{topic}'. Review and edit the blog post completed by the writer agent.
Make necessary corrections and improvements to ensure clarity and quality.
Once editing is complete, call the function to complete the blog post."""

# Transfer functions with error handling


def transfer_to_planner(context_variables: Dict[str, Any] = None):
    """Transfers control to the Planner Agent with context."""
    try:
        if context_variables:
            planner_agent.short_memory.add(
                role="system",
                content=f"Context: {json.dumps(context_variables)}"
            )
            logger.debug(
                f"Added context to planner_agent: {context_variables}")
        return planner_agent
    except Exception as e:
        logger.error(f"Error in transfer_to_planner: {str(e)}")
        raise


def transfer_to_researcher(context_variables: Dict[str, Any] = None):
    """Transfers control to the Researcher Agent with context."""
    try:
        if context_variables:
            researcher_agent.short_memory.add(
                role="system",
                content=f"Context: {json.dumps(context_variables)}"
            )
            logger.debug(
                f"Added context to researcher_agent: {context_variables}")
        return researcher_agent
    except Exception as e:
        logger.error(f"Error in transfer_to_researcher: {str(e)}")
        raise


def transfer_to_writer(context_variables: Dict[str, Any] = None):
    """Transfers control to the Writer Agent with context."""
    try:
        if context_variables:
            writer_agent.short_memory.add(
                role="system",
                content=f"Context: {json.dumps(context_variables)}"
            )
            logger.debug(f"Added context to writer_agent: {context_variables}")
        return writer_agent
    except Exception as e:
        logger.error(f"Error in transfer_to_writer: {str(e)}")
        raise


def transfer_to_editor(context_variables: Dict[str, Any] = None):
    """Transfers control to the Editor Agent with context."""
    try:
        if context_variables:
            editor_agent.short_memory.add(
                role="system",
                content=f"Context: {json.dumps(context_variables)}"
            )
            logger.debug(f"Added context to editor_agent: {context_variables}")
        return editor_agent
    except Exception as e:
        logger.error(f"Error in transfer_to_editor: {str(e)}")
        raise


def transfer_to_admin(context_variables: Dict[str, Any] = None):
    """Transfers control back to the Admin Agent with context."""
    try:
        if context_variables:
            admin_agent.short_memory.add(
                role="system",
                content=f"Context: {json.dumps(context_variables)}"
            )
            logger.debug(f"Added context to admin_agent: {context_variables}")
        return admin_agent
    except Exception as e:
        logger.error(f"Error in transfer_to_admin: {str(e)}")
        raise


def complete_blog_post(title: str, content: str, context_variables: Dict[str, Any] = None) -> str:
    """Saves the completed blog post to a markdown file in the output directory."""
    try:
        filename = title.lower().replace(" ", "-") + ".md"
        output_path = os.path.join(OUTPUT_DIR, filename)
        formatted_content = f"# {title}\n\n{content}\n\n*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(formatted_content)
        logger.debug(f"Blog post '{title}' written to {output_path}")
        return "Task completed"
    except Exception as e:
        logger.error(f"Error in complete_blog_post: {str(e)}")
        raise


# Agent initialization without tool_schema
admin_agent = Agent(
    model_name="ollama/llama3.2",
    agent_name="Admin Agent",
    system_prompt=admin_prompt,
    tools=[transfer_to_planner],
    max_loops=1,
    verbose=True,
    artifacts_on=True,
    artifacts_output_path=OUTPUT_DIR,
    artifacts_file_extension=".md",
    context_length=2048,
    autosave=True,
    saved_state_path=os.path.join(OUTPUT_DIR, "admin_agent_state.json")
)

planner_agent = Agent(
    model_name="ollama/llama3.2",
    agent_name="Planner Agent",
    system_prompt=planner_prompt,
    tools=[transfer_to_researcher],
    max_loops=3,
    dynamic_loops=True,
    verbose=True,
    artifacts_on=True,
    artifacts_output_path=OUTPUT_DIR,
    artifacts_file_extension=".md",
    context_length=2048,
    autosave=True,
    saved_state_path=os.path.join(OUTPUT_DIR, "planner_agent_state.json"),
    plan_enabled=True,
    planning_prompt="Create a detailed outline for the blog post"
)

researcher_agent = Agent(
    model_name="ollama/llama3.2",
    agent_name="Researcher Agent",
    system_prompt=researcher_prompt,
    tools=[transfer_to_writer],
    max_loops=3,
    dynamic_loops=True,
    verbose=True,
    artifacts_on=True,
    artifacts_output_path=OUTPUT_DIR,
    artifacts_file_extension=".md",
    context_length=2048,
    autosave=True,
    saved_state_path=os.path.join(OUTPUT_DIR, "researcher_agent_state.json")
)

writer_agent = Agent(
    model_name="ollama/llama3.2",
    agent_name="Writer Agent",
    system_prompt=writer_prompt,
    tools=[transfer_to_editor],
    max_loops=1,
    verbose=True,
    artifacts_on=True,
    artifacts_output_path=OUTPUT_DIR,
    artifacts_file_extension=".md",
    context_length=2048,
    autosave=True,
    saved_state_path=os.path.join(OUTPUT_DIR, "writer_agent_state.json"),
    dynamic_temperature_enabled=True
)

editor_agent = Agent(
    model_name="ollama/llama3.2",
    agent_name="Editor Agent",
    system_prompt=editor_prompt,
    tools=[complete_blog_post],
    max_loops=1,
    verbose=True,
    artifacts_on=True,
    artifacts_output_path=OUTPUT_DIR,
    artifacts_file_extension=".md",
    context_length=2048,
    autosave=True,
    saved_state_path=os.path.join(OUTPUT_DIR, "editor_agent_state.json")
)


def get_sample_topics():
    return [
        "The Impact of Artificial Intelligence on Modern Healthcare",
        "10 Tips for Remote Work Productivity",
        "How to Start Investing in Cryptocurrency",
        "The Future of Renewable Energy Technologies",
        "A Beginner's Guide to Mindfulness Meditation",
        "The Rise of Electric Vehicles: What You Need to Know",
        "How Social Media is Changing the News Industry",
        "The Benefits of Learning a Second Language",
        "Cybersecurity Best Practices for Small Businesses",
        "Exploring the World of Plant-Based Diets"
    ]


def select_topic():
    time.sleep(0.01)
    sample_topics = get_sample_topics()
    print("Sample Blog Topics:")
    for idx, topic in enumerate(sample_topics, 1):
        print(f"{idx}. {topic}")
    print("0. Enter your own topic")
    try:
        choice = int(
            input("Select a topic by number (or 0 to enter your own): "))
    except ValueError:
        choice = -1
    if 1 <= choice <= len(sample_topics):
        return sample_topics[choice - 1]
    elif choice == 0:
        return input("Please provide a topic for the blog post: ")
    else:
        print("Invalid selection. Please try again.")
        return select_topic()


def run():
    topic = select_topic()
    if not topic or not isinstance(topic, str):
        logger.error("Invalid or missing topic")
        raise ValueError("Topic must be a non-empty string")
    context_variables = {"topic": topic}
    current_agent = admin_agent
    while current_agent:
        try:
            # Format the system prompt with the current topic
            formatted_prompt = current_agent.system_prompt.format(
                topic=context_variables.get("topic", "No topic provided")
            )
            logger.debug(
                f"Formatted prompt for {current_agent.agent_name}: {formatted_prompt}")
            current_agent.set_system_prompt(formatted_prompt)
            context_str = f"Context: {json.dumps(context_variables)}"
            if current_agent.check_available_tokens() < len(current_agent.tokenizer.encode(context_str)):
                context_str = current_agent.truncate_string_by_tokens(
                    context_str, current_agent.context_length - 100)
                logger.debug(
                    f"Truncated context for {current_agent.agent_name}: {context_str}")
            current_agent.short_memory.add(
                role="system",
                content=context_str
            )
            logger.debug(
                f"Running {current_agent.agent_name} with tools: {[tool.__name__ for tool in current_agent.tools]}")
            result = current_agent.run(
                task="Process the blog post",
                context_variables=context_variables
            )
            logger.debug(f"Result from {current_agent.agent_name}: {result}")
            if isinstance(result, Agent):
                current_agent = result
            else:
                logger.info(f"Blog creation process completed: {result}")
                break
        except Exception as e:
            logger.error(
                f"Error in agent {current_agent.agent_name}: {str(e)}")
            raise
    logger.info("\n\n[DONE]")


if __name__ == "__main__":
    run()
