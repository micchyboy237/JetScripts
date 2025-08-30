async def main():
    from jet.transformers.formatters import format_json
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.core.agent.workflow import (
    AgentInput,
    AgentOutput,
    ToolCall,
    ToolCallResult,
    AgentStream,
    )
    from llama_index.core.agent.workflow import FunctionAgent, ReActAgent
    from llama_index.core.workflow import Context
    from tavily import AsyncTavilyClient
    import os
    import re
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    LOG_DIR = f"{OUTPUT_DIR}/logs"
    
    log_file = os.path.join(LOG_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.orange(f"Logs: {log_file}")
    
    """
    # Multi-Agent Report Generation using Agents as Tools
    
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/agent/agents_as_tools.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    In this notebook, we will explore how to create a multi-agent system that uses a top-level agent to orchestrate a group of agents as tools. Specifically, we will create a system that can research, write, and review a report on a given topic.
    
    This notebook will assume that you have already either read the [basic agent workflow notebook](https://docs.llamaindex.ai/en/stable/examples/agent/agent_workflow_basic) or the [general agent documentation](https://docs.llamaindex.ai/en/stable/understanding/agent/).
    
    ## Setup
    
    In this example, we will use `OllamaFunctionCallingAdapter` as our LLM. For all LLMs, check out the [examples documentation](https://docs.llamaindex.ai/en/stable/examples/llm/openai/) or [LlamaHub](https://llamahub.ai/?tab=llms) for a list of all supported LLMs and how to install/use them.
    
    If we wanted, each agent could have a different LLM, but for this example, we will use the same LLM for all agents.
    """
    logger.info("# Multi-Agent Report Generation using Agents as Tools")
    
    # %pip install llama-index
    
    
    sub_agent_llm = OllamaFunctionCallingAdapter(model="llama3.2", log_dir=f"{LOG_DIR}/chats")
    orchestrator_llm = OllamaFunctionCallingAdapter(model="o3-mini")
    
    """
    ## System Design
    
    Our system will have three agents:
    
    1. A `ResearchAgent` that will search the web for information on the given topic.
    2. A `WriteAgent` that will write the report using the information found by the `ResearchAgent`.
    3. A `ReviewAgent` that will review the report and provide feedback.
    
    We will then use a top-level agent to orchestrate the other agents to write our report.
    
    While there are many ways to implement this system, in this case, we will use a single `web_search` tool to search the web for information on the given topic.
    """
    logger.info("## System Design")
    
    # %pip install tavily-python
    
    
    
    async def search_web(query: str) -> str:
        """Useful for using the web to answer questions."""
        client = AsyncTavilyClient()
        return str(await client.search(query))
    
    """
    With our tool defined, we can now create our sub-agents.
    
    If the LLM you are using supports tool calling, you can use the `FunctionAgent` class. Otherwise, you can use the `ReActAgent` class.
    """
    logger.info("With our tool defined, we can now create our sub-agents.")
    
    
    research_agent = FunctionAgent(
        system_prompt=(
            "You are the ResearchAgent that can search the web for information on a given topic and record notes on the topic. "
            "You should output notes on the topic in a structured format."
        ),
        llm=sub_agent_llm,
        tools=[search_web],
    )
    
    write_agent = FunctionAgent(
        system_prompt=(
            "You are the WriteAgent that can write a report on a given topic. "
            "Your report should be in a markdown format. The content should be grounded in the research notes. "
            "Return your markdown report surrounded by <report>...</report> tags."
        ),
        llm=sub_agent_llm,
        tools=[],
    )
    
    review_agent = FunctionAgent(
        system_prompt=(
            "You are the ReviewAgent that can review the write report and provide feedback. "
            "Your review should either approve the current report or request changes to be implemented."
        ),
        llm=sub_agent_llm,
        tools=[],
    )
    
    """
    With our sub-agents defined, we can then convert them into tools that can be used by the top-level agent.
    """
    logger.info("With our sub-agents defined, we can then convert them into tools that can be used by the top-level agent.")
    
    
    
    async def call_research_agent(ctx: Context, prompt: str) -> str:
        """Useful for recording research notes based on a specific prompt."""
        result = await research_agent.run(
                user_msg=f"Write some notes about the following: {prompt}"
            )
        logger.success(format_json(result))
    
        async with ctx.store.edit_state() as ctx_state:
                ctx_state["state"]["research_notes"].append(str(result))
            
        logger.success(format_json(result))
        return str(result)
    
    
    async def call_write_agent(ctx: Context) -> str:
        """Useful for writing a report based on the research notes or revising the report based on feedback."""
        async with ctx.store.edit_state() as ctx_state:
                notes = ctx_state["state"].get("research_notes", None)
                if not notes:
                    return "No research notes to write from."
            
                user_msg = f"Write a markdown report from the following notes. Be sure to output the report in the following format: <report>...</report>:\n\n"
            
                feedback = ctx_state["state"].get("review", None)
                if feedback:
                    user_msg += f"<feedback>{feedback}</feedback>\n\n"
            
                notes = "\n\n".join(notes)
                user_msg += f"<research_notes>{notes}</research_notes>\n\n"
            
                result = await write_agent.run(user_msg=user_msg)
                report = re.search(
                    r"<report>(.*)</report>", str(result), re.DOTALL
                ).group(1)
                ctx_state["state"]["report_content"] = str(report)
            
        logger.success(format_json(result))
        return str(report)
    
    
    async def call_review_agent(ctx: Context) -> str:
        """Useful for reviewing the report and providing feedback."""
        async with ctx.store.edit_state() as ctx_state:
                report = ctx_state["state"].get("report_content", None)
                if not report:
                    return "No report content to review."
            
                result = await review_agent.run(
                    user_msg=f"Review the following report: {report}"
                )
                ctx_state["state"]["review"] = result
            
        logger.success(format_json(result))
        return result
    
    """
    ## Creating the Top-Level Orchestrator Agent
    
    With our sub-agents defined as tools, we can now create our top-level orchestrator agent.
    """
    logger.info("## Creating the Top-Level Orchestrator Agent")
    
    orchestrator = FunctionAgent(
        system_prompt=(
            "You are an expert in the field of report writing. "
            "You are given a user request and a list of tools that can help with the request. "
            "You are to orchestrate the tools to research, write, and review a report on the given topic. "
            "Once the review is positive, you should notify the user that the report is ready to be accessed."
        ),
        llm=orchestrator_llm,
        tools=[
            call_research_agent,
            call_write_agent,
            call_review_agent,
        ],
        initial_state={
            "research_notes": [],
            "report_content": None,
            "review": None,
        },
    )
    
    """
    ## Running the Agent
    
    Let's run our agents! We can iterate over events as the workflow runs.
    """
    logger.info("## Running the Agent")
    
    
    ctx = Context(orchestrator)
    
    
    async def run_orchestrator(ctx: Context, user_msg: str):
        handler = orchestrator.run(
            user_msg=user_msg,
            ctx=ctx,
        )
    
        async for event in handler.stream_events():
            if isinstance(event, AgentStream):
                if event.delta:
                    logger.debug(event.delta, end="", flush=True)
            elif isinstance(event, AgentOutput):
                if event.tool_calls:
                    logger.debug(
                        "🛠️  Planning to use tools:",
                        [call.tool_name for call in event.tool_calls],
                    )
            elif isinstance(event, ToolCallResult):
                logger.debug(f"🔧 Tool Result ({event.tool_name}):")
                logger.debug(f"  Arguments: {event.tool_kwargs}")
                logger.debug(f"  Output: {event.tool_output}")
            elif isinstance(event, ToolCall):
                logger.debug(f"🔨 Calling Tool: {event.tool_name}")
                logger.debug(f"  With arguments: {event.tool_kwargs}")
    
    await run_orchestrator(
        ctx=ctx,
        user_msg=(
            "Write me a report on the history of the internet. "
            "Briefly describe the history of the internet, including the development of the internet, the development of the web, "
            "and the development of the internet in the 21st century."
        ),
    )
    
    """
    With our report written and revised/reviewed, we can inspect the final report in the state.
    """
    logger.info("With our report written and revised/reviewed, we can inspect the final report in the state.")
    
    state = await ctx.store.get("state")
    logger.success(format_json(state))
    logger.debug(state["report_content"])
    
    logger.info("\n\n[DONE]", bright=True)

if __name__ == '__main__':
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())