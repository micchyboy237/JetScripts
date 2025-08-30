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
    from llama_index.core.agent.workflow import AgentWorkflow
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.workflow import Context
    from tavily import AsyncTavilyClient
    import os
    import shutil

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    LOG_DIR = f"{OUTPUT_DIR}/logs"

    log_file = os.path.join(LOG_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.orange(f"Logs: {log_file}")

    logger.info("# Multi-Agent Report Generation with AgentWorkflow")

    llm = OllamaFunctionCallingAdapter(
        model="llama3.2", log_dir=f"{LOG_DIR}/chats")

    logger.info("## System Design")

    async def search_web(query: str) -> str:
        """Useful for using the web to answer questions."""
        client = AsyncTavilyClient()
        search_results = await client.search(query)
        return str(search_results)

    async def record_notes(ctx: Context, notes: str, notes_title: str) -> str:
        """Useful for recording notes on a given topic. Your input should be notes with a title to save the notes under."""
        try:
            async with ctx.store.edit_state() as ctx_state:
                if "research_notes" not in ctx_state["state"]:
                    ctx_state["state"]["research_notes"] = {}
                ctx_state["state"]["research_notes"][notes_title] = notes
            result = {"notes_title": notes_title, "notes": notes}
            logger.success(format_json(result))
            return "Notes recorded."
        except Exception as e:
            logger.error(f"Error in record_notes: {str(e)}")
            raise

    async def write_report(ctx: Context, report_content: str) -> str:
        """Useful for writing a report on a given topic. Your input should be a markdown formatted report."""
        try:
            async with ctx.store.edit_state() as ctx_state:
                ctx_state["state"]["report_content"] = report_content
            result = {"report_content": report_content}
            logger.success(format_json(result))
            return "Report written."
        except Exception as e:
            logger.error(f"Error in write_report: {str(e)}")
            raise

    async def review_report(ctx: Context, review: str) -> str:
        """Useful for reviewing a report and providing feedback. Your input should be a review of the report."""
        try:
            async with ctx.store.edit_state() as ctx_state:
                ctx_state["state"]["review"] = review
            result = {"review": review}
            logger.success(format_json(result))
            return "Report reviewed."
        except Exception as e:
            logger.error(f"Error in review_report: {str(e)}")
            raise

    logger.info("With our tools defined, we can now create our agents.")

    # Updated ResearchAgent with stricter prompt
    research_agent = FunctionAgent(
        name="ResearchAgent",
        description="Useful for searching the web for information on a given topic.",
        system_prompt=(
            "You are the ResearchAgent responsible for searching the web for information on a given topic. "
            "Your task is strictly tool-based: "
            "1. If no search has been done, call the 'search_web' tool with the user's query. "
            "2. After receiving the 'tool' response from 'search_web', do NOT output any text content, summary, or response. "
            "Instead, immediately call the 'handoff' tool with 'to_agent' set to 'RecordAgent' and 'reason' set to a brief description like 'Search results obtained; handing off for note recording.' "
            "Never generate text in your response—always respond only with tool calls. Violation of this will terminate the workflow prematurely."
        ),
        llm=llm,
        tools=[search_web],
        can_handoff_to=["RecordAgent"],
    )

    record_agent = FunctionAgent(
        name="RecordAgent",
        description="Useful for recording notes on a given topic.",
        system_prompt=(
            "You are the RecordAgent. Retrieve search results from the workflow state ('search_results'). "
            "Summarize them into concise notes (without outputting text directly). "
            "Call the 'record_notes' tool with 'notes' as the summary and 'notes_title' as a suitable title. "
            "After calling 'record_notes', immediately call the 'handoff' tool with 'to_agent' set to 'WriteAgent' and 'reason' like 'Notes recorded; handing off for report writing.' "
            "Respond only with tool calls—no text content."
        ),
        llm=llm,
        tools=[record_notes],
        can_handoff_to=["WriteAgent"],
    )

    write_agent = FunctionAgent(
        name="WriteAgent",
        description="Useful for writing a report on a given topic.",
        system_prompt=(
            "You are the WriteAgent responsible for writing a report in markdown format based on the research notes. "
            "Use the 'write_report' tool to save the report content. "
            "After writing the report, always hand off control to the ReviewAgent for feedback."
        ),
        llm=llm,
        tools=[write_report],
        can_handoff_to=["ReviewAgent"],
    )

    review_agent = FunctionAgent(
        name="ReviewAgent",
        description="Useful for reviewing a report and providing feedback.",
        system_prompt=(
            "You are the ReviewAgent that can review the written report and provide feedback. "
            "Your review should either approve the current report or request changes for the WriteAgent to implement. "
            "If you have feedback that requires changes, you should hand off control to the WriteAgent to implement the changes after submitting the review."
        ),
        llm=llm,
        tools=[review_report],
        can_handoff_to=["WriteAgent"],
    )

    logger.info("## Running the Workflow")

    agent_workflow = AgentWorkflow(
        agents=[research_agent, record_agent, write_agent, review_agent],
        root_agent=research_agent.name,
        initial_state={
            "research_notes": {},
            "report_content": "Not written yet.",
            "review": "Review required.",
            "search_results": None,
        },
    )

    logger.info(
        "As the workflow is running, we will stream the events to get an idea of what is happening under the hood.")

    handler = agent_workflow.run(
        user_msg=(
            "Write me a report on the history of the internet. "
            "Briefly describe the history of the internet, including the development of the internet, the development of the web, "
            "and the development of the internet in the 21st century."
        )
    )

    current_agent = None
    current_tool_calls = ""
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            logger.debug(f"\n{'='*50}")
            logger.debug(f"🤖 Agent: {current_agent}")
            logger.debug(f"{'='*50}\n")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                logger.debug("📤 Output:", event.response.content)
            if event.tool_calls:
                logger.debug("🛠️ Planning to use tools:", [
                             call.tool_name for call in event.tool_calls])
                for tool_call in event.tool_calls:
                    logger.debug(f"🔧 Tool Call: {tool_call.tool_name}")
                    logger.debug(f"  Arguments: {tool_call.tool_kwargs}")
                    if tool_call.tool_name == "handoff":
                        logger.debug(
                            f"🤝 Handoff to: {tool_call.tool_kwargs.get('to_agent', 'Unknown')}")
        elif isinstance(event, ToolCallResult):
            logger.debug(f"🔧 Tool Result ({event.tool_name}):")
            logger.debug(f"  Arguments: {event.tool_kwargs}")
            logger.debug(f"  Output: {event.tool_output}")
            if event.tool_name == "search_web":
                async with handler.ctx.store.edit_state() as ctx_state:
                    ctx_state["state"]["search_results"] = event.tool_output
        elif isinstance(event, ToolCall):
            logger.debug(f"🔨 Calling Tool: {event.tool_name}")
            logger.debug(f"  With arguments: {event.tool_kwargs}")

    logger.info(
        "Now, we can retrieve the final report in the system for ourselves.")

    state = await handler.ctx.store.get("state")
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
