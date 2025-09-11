from jet.transformers.formatters import format_json
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_hyperbrowser import HyperbrowserBrowserUseTool
from langchain_hyperbrowser import HyperbrowserClaudeComputerUseTool
from langchain_hyperbrowser import HyperbrowserOllamaCUATool
from langchain_hyperbrowser import browser_use_tool
from langgraph.prebuilt import create_react_agent
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
---
sidebar_label: Hyperbrowser Browser Agent Tools
---

# Hyperbrowser Browser Agent Tools

[Hyperbrowser](https://hyperbrowser.ai) is a platform for running, running browser agents, and scaling headless browsers. It lets you launch and manage browser sessions at scale and provides easy to use solutions for any webscraping needs, such as scraping a single page or crawling an entire site.

Key Features:
- Instant Scalability - Spin up hundreds of browser sessions in seconds without infrastructure headaches
- Simple Integration - Works seamlessly with popular tools like Puppeteer and Playwright
- Powerful APIs - Easy to use APIs for scraping/crawling any site, and much more
- Bypass Anti-Bot Measures - Built-in stealth mode, ad blocking, automatic CAPTCHA solving, and rotating proxies

This notebook provides a quick overview for getting started with Hyperbrowser tools.

For more information about Hyperbrowser, please visit the [Hyperbrowser website](https://hyperbrowser.ai) or if you want to check out the docs, you can visit the [Hyperbrowser docs](https://docs.hyperbrowser.ai).


## Browser Agents

Hyperbrowser provides powerful browser agent tools that enable AI models to interact with web browsers programmatically. These browser agents can navigate websites, fill forms, click buttons, extract data, and perform complex web automation tasks.

Browser agents are particularly useful for:
- Web scraping and data extraction from complex websites
- Automating repetitive web tasks
- Interacting with web applications that require authentication
- Performing research across multiple websites
- Testing web applications

Hyperbrowser offers three types of browser agent tools:
- **Browser Use Tool**: A general-purpose browser automation tool
- **Ollama CUA Tool**: Integration with Ollama's Computer Use Agent
- **Claude Computer Use Tool**: Integration with Ollama's Claude for computer use


## Overview

### Integration details

| Tool                      | Package                | Local | Serializable | JS support |
| :-----------------------  | :--------------------- | :---: | :----------: | :--------: |
| Browser Use Tool          | langchain-hyperbrowser |  ❌   |      ❌      |      ❌    |
| Ollama CUA Tool           | langchain-hyperbrowser |  ❌   |      ❌      |      ❌    |
| Claude Computer Use Tool  | langchain-hyperbrowser |  ❌   |      ❌      |     ❌     |

## Setup

To access the Hyperbrowser tools you'll need to install the `langchain-hyperbrowser` integration package, and create a Hyperbrowser account and get an API key.

### Credentials

Head to [Hyperbrowser](https://app.hyperbrowser.ai/) to sign up and generate an API key. Once you've done this set the HYPERBROWSER_API_KEY environment variable:

```bash
export HYPERBROWSER_API_KEY=<your-api-key>
```

### Installation

Install **langchain-hyperbrowser**.
"""
logger.info("# Hyperbrowser Browser Agent Tools")

# %pip install -qU langchain-hyperbrowser

"""
## Instantiation

### Browser Use Tool

The `HyperbrowserBrowserUseTool` is a tool to perform web automation tasks using a browser agent, specifically the Browser-Use agent.

```python
tool = HyperbrowserBrowserUseTool()
```

### Ollama CUA Tool

The `HyperbrowserOllamaCUATool` is a specialized tool that leverages Ollama's Computer Use Agent (CUA) capabilities through Hyperbrowser.

```python
tool = HyperbrowserOllamaCUATool()
```

### Claude Computer Use Tool

The `HyperbrowserClaudeComputerUseTool` is a specialized tool that leverages Claude's computer use capabilities through Hyperbrowser.

```python
tool = HyperbrowserClaudeComputerUseTool()
```

## Invocation

### Basic Usage

#### Browser Use Tool
"""
logger.info("## Instantiation")


tool = HyperbrowserBrowserUseTool()
result = tool.run({"task": "Go to Hacker News and summarize the top 5 posts right now"})
logger.debug(result)

"""
#### Ollama CUA Tool
"""
logger.info("#### Ollama CUA Tool")


tool = HyperbrowserOllamaCUATool()
result = tool.run(
    {"task": "Go to Hacker News and get me the title of the top 5 posts right now"}
)
logger.debug(result)

"""
#### Claude Computer Use Tool
"""
logger.info("#### Claude Computer Use Tool")


tool = HyperbrowserClaudeComputerUseTool()
result = tool.run({"task": "Go to Hacker News and summarize the top 5 posts right now"})
logger.debug(result)

"""
### With Custom Session Options

All tools support custom session options:
"""
logger.info("### With Custom Session Options")

result = tool.run(
    {
        "task": "Go to npmjs.com, and tell me when react package was last updated.",
        "session_options": {
            "session_options": {"use_proxy": True, "accept_cookies": True}
        },
    }
)
logger.debug(result)

"""
### Async Usage

All tools support async usage:
"""
logger.info("### Async Usage")

async def browse_website():
    tool = HyperbrowserBrowserUseTool()
    result = await tool.arun(
            {
                "task": "Go to npmjs.com, click the first visible package, and tell me when it was updated"
            }
        )
    logger.success(format_json(result))
    return result


result = await browse_website()
logger.success(format_json(result))

"""
## Use within an agent

Here's how to use any of the Hyperbrowser tools within an agent:
"""
logger.info("## Use within an agent")


llm = ChatOllama(model="llama3.2")

browser_use_tool = HyperbrowserBrowserUseTool()
agent = create_react_agent(llm, [browser_use_tool])

user_input = "Go to npmjs.com, and tell me when react package was last updated."
for step in agent.stream(
    {"messages": user_input},
    stream_mode="values",
):
    step["messages"][-1].pretty_logger.debug()

"""
## Configuration Options

Claude Computer Use, Ollama CUA, and Browser Use have the following params available:

- `task`: The task to execute using the agent
- `max_steps`: The maximum number of interaction steps the agent can take to complete the task
- `session_options`: Browser session configuration

For more details, see the respective API references:
- [Browser Use API Reference](https://docs.hyperbrowser.ai/reference/api-reference/agents/browser-use)
- [Ollama CUA API Reference](https://docs.hyperbrowser.ai/reference/api-reference/agents/ollama-cua)
- [Claude Computer Use API Reference](https://docs.hyperbrowser.ai/reference/api-reference/agents/claude-computer-use)

## API reference

- [GitHub](https://github.com/hyperbrowserai/langchain-hyperbrowser/)
- [PyPi](https://pypi.org/project/langchain-hyperbrowser/)
- [Hyperbrowser Docs](https://docs.hyperbrowser.ai/)
"""
logger.info("## Configuration Options")

logger.info("\n\n[DONE]", bright=True)