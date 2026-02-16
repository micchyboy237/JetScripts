# jet_python_modules/jet/libs/smolagents/jet_examples/search_tool_searxng.py
import shutil
from pathlib import Path

from jet.libs.smolagents.tools.searxng_search_tool2 import SearXNGSearchTool
from jet.libs.smolagents.utils import create_local_model
from smolagents import LogLevel, ToolCallingAgent

OUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUT_DIR, ignore_errors=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

model = create_local_model()


agent = ToolCallingAgent(
    tools=[SearXNGSearchTool(max_results=10)],
    model=model,
    add_base_tools=False,
    verbosity_level=LogLevel.DEBUG,  # ← Add this line (or LogLevel.INFO)
)

if __name__ == "__main__":
    result = agent.run("What's the weather like in Las Piñas City right now?")
    print(result)
