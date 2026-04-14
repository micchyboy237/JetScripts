from jet.libs.smolagents.tools.searxng_search_tool import (
    SearXNGSearchTool,  # or any from jet
)
from jet.libs.smolagents.utils.model_utils import (
    create_local_model,  # reuse your jet module
)
from smolagents import CodeAgent, ToolCallingAgent

# Assume VisitWebpageTool or your custom @tool from previous files

# 1. Create stateless sub-agent (ToolCallingAgent is great for single-timeline tasks)
web_sub_agent = ToolCallingAgent(
    tools=[
        SearXNGSearchTool(max_results=10),
        # VisitWebpageTool(...) or your custom visit_webpage @tool
    ],
    model=create_local_model(temperature=0.65, agent_name="web_sub_agent"),
    max_steps=10,
    name="web_agent",  # ← Required for delegation
    description="Performs web searches and visits pages to gather detailed information.",
    verbosity_level=1,
)

# 2. Create manager (CodeAgent) that can delegate to the sub-agent
manager = CodeAgent(
    tools=[],  # manager can have its own tools if needed
    model=create_local_model(temperature=0.7, agent_name="manager_agent"),
    managed_agents=[web_sub_agent],  # ← sub-agents become callable "tools"
    max_steps=15,
    verbosity_level=2,
    additional_authorized_imports=["time", "numpy", "pandas"],  # for calculations
)

# 3. Run – manager decides when to call the stateless sub-agent
task = (
    "Find the latest stable version of Hugging Face Transformers library. "
    "Then estimate how many parameters the next major release might have based on trends."
)
answer = manager.run(task, reset=True)  # reset=True for fresh stateless start
print(answer)
