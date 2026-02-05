# demo_simple_nav.py
"""
Minimal demo using smolagents + NavigationalSearchTool with embeddings
"""

from jet.libs.smolagents.text_web_browser import SimpleTextBrowser
from jet.libs.smolagents.tools.text_web_browser.navigational_search_tool import (
    NavigationalSearchTool,
)
from smolagents import Agent, OpenAIModel

# ────────────────────────────────────────────────────────────────────────────────
#  Choose your model (must support tool calling)
# ────────────────────────────────────────────────────────────────────────────────

model = OpenAIModel(
    model="gpt-4o-mini",  # or gpt-4o, o1-mini, etc.
    temperature=0.7,
)

# ────────────────────────────────────────────────────────────────────────────────
#  Tools
# ────────────────────────────────────────────────────────────────────────────────

browser = SimpleTextBrowser()  # the main browser
nav_tool = NavigationalSearchTool(browser)  # our improved navigational helper

tools = [browser, nav_tool]

# ────────────────────────────────────────────────────────────────────────────────
#  Create agent
# ────────────────────────────────────────────────────────────────────────────────

agent = Agent(
    model=model,
    tools=tools,
    system_prompt=(
        "You are a helpful web navigation assistant.\n"
        "Use the navigational_search tool to suggest the most promising next links "
        "when you need to decide where to go next.\n"
        "Use browse_page / page_content when you need to read actual content.\n"
        "Be concise and goal-directed."
    ),
)

# ────────────────────────────────────────────────────────────────────────────────
#  Example interaction loop
# ────────────────────────────────────────────────────────────────────────────────


def run_demo():
    print("Starting web navigation demo with embeddings-based link ranking\n")
    initial_url = "https://docs.python.org/3/"
    print(f"→ Starting at: {initial_url}\n")

    browser.address = initial_url
    browser.refresh()  # or .goto(initial_url) depending on exact API

    goal = "Find information about how to install Python packages using pip"

    print("Goal:", goal, "\n")

    response = agent.run(
        f"Current page: {browser.address}\n"
        f"Goal: {goal}\n"
        "Suggest the 2–3 most promising next links using navigational_search. "
        "Explain your reasoning briefly."
    )

    print("\nAgent suggestion:\n")
    print(response)


if __name__ == "__main__":
    run_demo()
