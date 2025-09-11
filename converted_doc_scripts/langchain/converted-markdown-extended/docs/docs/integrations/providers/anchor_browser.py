from jet.logger import logger
from langchain_anchorbrowser import AnchorContentTool
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# Anchor Browser

[Anchor](https://anchorbrowser.io?utm=langchain) is the platform for AI Agentic browser automation, which solves the challenge of automating workflows for web applications that lack APIs or have limited API coverage. It simplifies the creation, deployment, and management of browser-based automations, transforming complex web interactions into simple API endpoints.

`langchain-anchorbrowser` provides 3 main tools:
- `AnchorContentTool` - For web content extractions in Markdown or HTML format.
- `AnchorScreenshotTool` - For web page screenshots.
- `AnchorWebTaskTools` - To perform web tasks.

## Quickstart

### Installation

Install the package:
"""
logger.info("# Anchor Browser")

pip install langchain-anchorbrowser

"""
### Usage

Import and utilize your intended tool. The full list of Anchor Browser available tools see **Tool Features** table in [Anchor Browser tool page](/docs/integrations/tools/anchor_browser)
"""
logger.info("### Usage")


AnchorContentTool().invoke(
    {"url": "https://www.anchorbrowser.io", "format": "markdown"}
)

"""
## Additional Resources

- [PyPi](https://pypi.org/project/langchain-anchorbrowser)
- [Github](https://github.com/anchorbrowser/langchain-anchorbrowser)
- [Anchor Browser Docs](https://docs.anchorbrowser.io/introduction?utm=langchain)
- [Anchor Browser API Reference](https://docs.anchorbrowser.io/api-reference/ai-tools/perform-web-task?utm=langchain)
"""
logger.info("## Additional Resources")

logger.info("\n\n[DONE]", bright=True)