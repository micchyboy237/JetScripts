from jet.logger import logger
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
# Agents

By themselves, language models can't take actions - they just output text. Agents are systems that take a high-level task and use an LLM as a reasoning engine to decide what actions to take and execute those actions.

[LangGraph](/docs/concepts/architecture#langgraph) is an extension of LangChain specifically aimed at creating highly controllable and customizable agents. We recommend that you use LangGraph for building agents.

Please see the following resources for more information:

* LangGraph docs on [common agent architectures](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/)
* [Pre-built agents in LangGraph](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent)

## Legacy agent concept: AgentExecutor

LangChain previously introduced the `AgentExecutor` as a runtime for agents.
While it served as an excellent starting point, its limitations became apparent when dealing with more sophisticated and customized agents.
As a result, we're gradually phasing out `AgentExecutor` in favor of more flexible solutions in LangGraph.

### Transitioning from AgentExecutor to LangGraph

If you're currently using `AgentExecutor`, don't worry! We've prepared resources to help you:

1. For those who still need to use `AgentExecutor`, we offer a comprehensive guide on [how to use AgentExecutor](/docs/how_to/agent_executor).

2. However, we strongly recommend transitioning to LangGraph for improved flexibility and control. To facilitate this transition, we've created a detailed [migration guide](/docs/how_to/migrate_agent) to help you move from `AgentExecutor` to LangGraph seamlessly.
"""
logger.info("# Agents")

logger.info("\n\n[DONE]", bright=True)