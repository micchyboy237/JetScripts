async def main():
    from jet.transformers.formatters import format_json
    from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
    from jet.logger import CustomLogger
    from llama_index.core.agent.workflow import FunctionAgent, ReActAgent
    from llama_index.core.llms import ChatMessage
    from llama_index.core.workflow import (
        Context,
        Event,
        StartEvent,
        StopEvent,
        Workflow,
        step,
    )
    from llama_index.core.workflow import Context
    from pydantic import BaseModel, Field
    from tavily import AsyncTavilyClient
    from typing import Any, Optional
    import os
    import re
    import shutil
    import xml.etree.ElementTree as ET

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    LOG_DIR = f"{OUTPUT_DIR}/logs"

    log_file = os.path.join(LOG_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.orange(f"Logs: {log_file}")

    """
    # Custom Planning Multi-Agent System
    
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/agent/custom_multi_agent.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    In this notebook, we will explore how to prompt an LLM to write, refine, and follow a plan to generate a report using multiple agents.
    
    This is not meant to be a comprehensive guide to creating a report generation system, but rather, giving you the knowledge and tools to build your own robust systems that can plan and orchestrate multiple agents to achieve a goal.
    
    This notebook will assume that you have already either read the [basic agent workflow notebook](https://docs.llamaindex.ai/en/stable/examples/agent/agent_workflow_basic) or the [agent workflow documentation](https://docs.llamaindex.ai/en/stable/understanding/agent/), as well as the [workflow documentation](https://docs.llamaindex.ai/en/stable/understanding/workflows/).
    
    ## Setup
    
    In this example, we will use `OllamaFunctionCalling` as our LLM. For all LLMs, check out the [examples documentation](https://docs.llamaindex.ai/en/stable/examples/llm/openai/) or [LlamaHub](https://llamahub.ai/?tab=llms) for a list of all supported LLMs and how to install/use them.
    
    If we wanted, each agent could have a different LLM, but for this example, we will use the same LLM for all agents.
    """
    logger.info("# Custom Planning Multi-Agent System")

    # %pip install llama-index

    sub_agent_llm = OllamaFunctionCalling(
        model="llama3.2", log_dir=f"{LOG_DIR}/chats")

    """
    ## System Design
    
    Our system will have three agents:
    
    1. A `ResearchAgent` that will search the web for information on the given topic.
    2. A `WriteAgent` that will write the report using the information found by the `ResearchAgent`.
    3. A `ReviewAgent` that will review the report and provide feedback.
    
    We will then use a top-level LLM to manually orchestrate and plan around the other agents to write our report.
    
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
        name="ResearchAgent",
        description="Useful for recording research notes based on a specific prompt.",
        system_prompt=(
            "You are the ResearchAgent that can search the web for information on a given topic and record notes on the topic. "
            "You should output notes on the topic in a structured format."
        ),
        llm=sub_agent_llm,
        tools=[search_web],
    )

    write_agent = FunctionAgent(
        name="WriteAgent",
        description="Useful for writing a report based on the research notes or revising the report based on feedback.",
        system_prompt=(
            "You are the WriteAgent that can write a report on a given topic. "
            "Your report should be in a markdown format. The content should be grounded in the research notes. "
            "Return your markdown report surrounded by <report>...</report> tags."
        ),
        llm=sub_agent_llm,
        tools=[],
    )

    review_agent = FunctionAgent(
        name="ReviewAgent",
        description="Useful for reviewing a report and providing feedback.",
        system_prompt=(
            "You are the ReviewAgent that can review the write report and provide feedback. "
            "Your review should either approve the current report or request changes to be implemented."
        ),
        llm=sub_agent_llm,
        tools=[],
    )

    """
    With each agent defined, we can also write helper functions to help execute each agent.
    """
    logger.info(
        "With each agent defined, we can also write helper functions to help execute each agent.")

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
    ## Defining the Planner Workflow
    
    In order to plan around the other agents, we will write a custom workflow that will explicitly orchestrate and plan the other agents.
    
    Here our prompt assumes a sequential plan, but we can expand it in the future to support parallel steps. (This just involves more complex parsing and prompting, which is left as an exercise for the reader.)
    """
    logger.info("## Defining the Planner Workflow")

    PLANNER_PROMPT = """You are a planner chatbot.
    
    Given a user request and the current state, break the solution into ordered <step> blocks.  Each step must specify the agent to call and the message to send, e.g.
    <plan>
      <step agent=\"ResearchAgent\">search for …</step>
      <step agent=\"WriteAgent\">draft a report …</step>
      ...
    </plan>
    
    <state>
    {state}
    </state>
    
    <available_agents>
    {available_agents}
    </available_agents>
    
    The general flow should be:
    - Record research notes
    - Write a report
    - Review the report
    - Write the report again if the review is not positive enough
    
    If the user request does not require any steps, you can skip the <plan> block and respond directly.
    """

    class InputEvent(StartEvent):
        user_msg: Optional[str] = Field(default=None)
        chat_history: list[ChatMessage]
        state: Optional[dict[str, Any]] = Field(default=None)

    class OutputEvent(StopEvent):
        response: str
        chat_history: list[ChatMessage]
        state: dict[str, Any]

    class StreamEvent(Event):
        delta: str

    class PlanEvent(Event):
        step_info: str

    class PlanStep(BaseModel):
        agent_name: str
        agent_input: str

    class Plan(BaseModel):
        steps: list[PlanStep]

    class ExecuteEvent(Event):
        plan: Plan
        chat_history: list[ChatMessage]

    class PlannerWorkflow(Workflow):
        llm: OllamaFunctionCalling = sub_agent_llm
        agents: dict[str, FunctionAgent] = {
            "ResearchAgent": research_agent,
            "WriteAgent": write_agent,
            "ReviewAgent": review_agent,
        }

        @step
        async def plan(
            self, ctx: Context, ev: InputEvent
        ) -> ExecuteEvent | OutputEvent:
            if ev.state:
                await ctx.store.set("state", ev.state)

            chat_history = ev.chat_history

            if ev.user_msg:
                user_msg = ChatMessage(
                    role="user",
                    content=ev.user_msg,
                )
                chat_history.append(user_msg)

            state = await ctx.store.get("state")
            logger.success(format_json(state))
            available_agents_str = "\n".join(
                [
                    f'<agent name="{agent.name}">{agent.description}</agent>'
                    for agent in self.agents.values()
                ]
            )
            system_prompt = ChatMessage(
                role="system",
                content=PLANNER_PROMPT.format(
                    state=str(state),
                    available_agents=available_agents_str,
                ),
            )

            # Remove await since llm.chat is not async
            response = self.llm.chat(
                messages=[system_prompt] + chat_history,
            )
            logger.success(format_json(response))
            full_response = str(response.message.content)

            # Simulate streaming by writing the full response to the stream
            ctx.write_event_to_stream(
                StreamEvent(delta=full_response),
            )

            xml_match = re.search(r"(<plan>.*</plan>)",
                                  full_response, re.DOTALL)

            if not xml_match:
                chat_history.append(
                    ChatMessage(
                        role="assistant",
                        content=full_response,
                    )
                )
                return OutputEvent(
                    response=full_response,
                    chat_history=chat_history,
                    state=state,
                )
            else:
                xml_str = xml_match.group(1)
                root = ET.fromstring(xml_str)
                plan = Plan(steps=[])
                for step in root.findall("step"):
                    plan.steps.append(
                        PlanStep(
                            agent_name=step.attrib["agent"],
                            agent_input=step.text.strip() if step.text else "",
                        )
                    )

                return ExecuteEvent(plan=plan, chat_history=chat_history)

        @step
        async def execute(self, ctx: Context, ev: ExecuteEvent) -> InputEvent:
            chat_history = ev.chat_history
            plan = ev.plan

            for step in plan.steps:
                agent = self.agents[step.agent_name]
                agent_input = step.agent_input
                ctx.write_event_to_stream(
                    PlanEvent(
                        step_info=f'<step agent="{step.agent_name}">{step.agent_input}</step>'
                    ),
                )

                if step.agent_name == "ResearchAgent":
                    await call_research_agent(ctx, agent_input)
                elif step.agent_name == "WriteAgent":
                    await call_write_agent(ctx)
                elif step.agent_name == "ReviewAgent":
                    await call_review_agent(ctx)

            state = await ctx.store.get("state")
            logger.success(format_json(state))
            chat_history.append(
                ChatMessage(
                    role="user",
                    content=f"I've completed the previous steps, here's the updated state:\n\n<state>\n{state}\n</state>\n\nDo you need to continue and plan more steps?, If not, write a final response.",
                )
            )

            return InputEvent(
                chat_history=chat_history,
            )

    """
    ## Running the Workflow
    
    With our custom planner defined, we can now run the workflow and see it in action!
    
    As the workflow is running, we will stream the events to get an idea of what is happening under the hood.
    """
    logger.info("## Running the Workflow")

    planner_workflow = PlannerWorkflow(timeout=None)

    handler = planner_workflow.run(
        user_msg=(
            "Write me a report on the history of the internet. "
            "Briefly describe the history of the internet, including the development of the internet, the development of the web, "
            "and the development of the internet in the 21st century."
        ),
        chat_history=[],
        state={
            "research_notes": [],
            "report_content": "Not written yet.",
            "review": "Review required.",
        },
    )

    current_agent = None
    current_tool_calls = ""
    async for event in handler.stream_events():
        if isinstance(event, PlanEvent):
            logger.debug("Executing plan step: ", event.step_info)
        elif isinstance(event, ExecuteEvent):
            logger.debug("Executing plan: ", event.plan)

    result = await handler
    logger.success(format_json(result))

    logger.debug(result.response)

    """
    Now, we can retrieve the final report in the system for ourselves.
    """
    logger.info(
        "Now, we can retrieve the final report in the system for ourselves.")

    state = await handler.ctx.store.get("state")
    logger.success(format_json(state))
    logger.debug(state["report_content"])

    logger.debug(state["review"])

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
