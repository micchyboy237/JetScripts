import os

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from agent_framework import Agent, tool
from agent_framework.devui import serve
from agent_framework.openai import OpenAIChatClient

# --- Configuration ---
LLM_BASE_URL = os.getenv("LLAMA_CPP_LLM_URL", "http://192.168.68.150:8080/v1")
LLM_MODEL = os.getenv("LLAMA_CPP_LLM_MODEL", "qwen3.5-uncensored:2b")
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888")

# --- Dynamic Tools ---


@tool
def search_local_web(query: str) -> str:
    """
    Search the local SearXNG instance for real-time information.
    Use this when you need up-to-date facts, news, or technical documentation.
    """
    try:
        # Dynamic request to your local search engine
        response = requests.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        if not data.get("results"):
            return f"No results found for '{query}'."

        # Dynamically format top 3 results
        results_summary = []
        for r in data["results"][:3]:
            title = r.get("title", "No Title")
            url = r.get("url", "No URL")
            content = r.get("content", "")[:100] + "..." if r.get("content") else ""
            results_summary.append(f"- {title}\n  URL: {url}\n  Snippet: {content}")

        return "\n\n".join(results_summary)
    except Exception as e:
        return f"Error accessing search engine: {str(e)}"


@tool
def calculate_dynamic_metric(value_a: float, value_b: float, operation: str) -> float:
    """
    Perform a dynamic calculation based on user input.
    Useful for financial projections, scientific data analysis, or live metrics.
    """
    if operation == "add":
        return value_a + value_b
    elif operation == "subtract":
        return value_a - value_b
    elif operation == "multiply":
        return value_a * value_b
    elif operation == "divide":
        if value_b == 0:
            return "Error: Division by zero"
        return value_a / value_b
    else:
        return f"Unknown operation: {operation}"


# --- Agent Definition ---

# Configure the client to point to your local Llama.cpp server
# IMPORTANT: For local LLMs like Llama.cpp, we must disable server-side storage
# to avoid sending 'previous_response_id' which is unsupported.
client = OpenAIChatClient(
    base_url=LLM_BASE_URL,
    model=LLM_MODEL,
    api_key="not-required",
)

# Create the agent with dynamic tools
agent = Agent(
    name="DynamicResearchAssistant",
    client=client,
    tools=[search_local_web, calculate_dynamic_metric],
    instructions=(
        "You are a research assistant with access to live web search and calculation tools. "
        "1. Use 'search_local_web' to find current information. "
        "2. Use 'calculate_dynamic_metric' for any math required. "
        "Always cite sources from search results. "
        "Do not make up facts; if search fails, admit it."
    ),
)

# --- Launch DevUI ---

if __name__ == "__main__":
    print(f"Starting DevUI for agent: {agent.name}")
    print(f"Connecting to LLM at: {LLM_BASE_URL}")
    print(f"Using Search Engine at: {SEARXNG_URL}")

    # Serve the agent locally
    # host="0.0.0.0" allows access from other devices on your network
    # auth_token is required for non-loopback hosts
    serve(
        entities=[agent],
        host="0.0.0.0",
        port=8090,
        auto_open=True,
        auth_token="my-secret-dev-token",
    )
