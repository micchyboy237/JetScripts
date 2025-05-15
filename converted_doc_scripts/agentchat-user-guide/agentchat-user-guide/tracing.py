import asyncio
from jet.transformers.formatters import format_json
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_core import SingleThreadedAgentRuntime
from autogen_ext.models.openai import OllamaChatCompletionClient
from jet.logger import CustomLogger
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Tracing and Observability

AutoGen has [built-in support for tracing](https://microsoft.github.io/autogen/dev/user-guide/core-user-guide/framework/telemetry.html) and observability for collecting comprehensive records on the execution of your application. This feature is useful for debugging, performance analysis, and understanding the flow of your application.

This capability is powered by the [OpenTelemetry](https://opentelemetry.io/) library, which means you can use any OpenTelemetry-compatible backend to collect and analyze traces.

## Setup

To begin, you need to install the OpenTelemetry Python package. You can do this using pip:

```bash
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc
```

Once you have the SDK installed, the simplest way to set up tracing in AutoGen is to:

1. Configure an OpenTelemetry tracer provider
2. Set up an exporter to send traces to your backend
3. Connect the tracer provider to the AutoGen runtime

## Telemetry Backend

To collect and view traces, you need to set up a telemetry backend. Several open-source options are available, including Jaeger, Zipkin. For this example, we will use Jaeger as our telemetry backend.

For a quick start, you can run Jaeger locally using Docker:

```bash
docker run -d --name jaeger \
  -e COLLECTOR_OTLP_ENABLED=true \
  -p 16686:16686 \
  -p 4317:4317 \
  -p 4318:4318 \
  jaegertracing/all-in-one:latest
```

This command starts a Jaeger instance that listens on port 16686 for the Jaeger UI and port 4317 for the OpenTelemetry collector. You can access the Jaeger UI at `http://localhost:16686`.

## Instrumenting an AgentChat Team

In the following section, we will review how to enable tracing with an AutoGen GroupChat team. The AutoGen runtime already supports open telemetry (automatically logging message metadata). To begin, we will create a tracing service that will be used to instrument the AutoGen runtime.
"""
logger.info("# Tracing and Observability")


otel_exporter = OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)
tracer_provider = TracerProvider(resource=Resource({"service.name": "autogen-test-agentchat"}))
span_processor = BatchSpanProcessor(otel_exporter)
tracer_provider.add_span_processor(span_processor)
trace.set_tracer_provider(tracer_provider)

"""
All of the code to create a [team](./tutorial/teams.ipynb) should already be familiar to you.  An important note here is that all AgentChat agents and teams are run using the AutoGen core API runtime. In turn, the runtime is already instrumented to log [runtime messaging events (metadata)] (https://github.com/microsoft/autogen/blob/main/python/packages/autogen-core/src/autogen_core/_telemetry/_tracing_config.py) including:

- **create**: When a message is created
- **send**: When a message is sent
- **publish**: When a message is published
- **receive**: When a message is received
- **intercept**: When a message is intercepted
- **process**: When a message is processed
- **ack**: When a message is acknowledged
"""
logger.info("All of the code to create a [team](./tutorial/teams.ipynb) should already be familiar to you.  An important note here is that all AgentChat agents and teams are run using the AutoGen core API runtime. In turn, the runtime is already instrumented to log [runtime messaging events (metadata)] (https://github.com/microsoft/autogen/blob/main/python/packages/autogen-core/src/autogen_core/_telemetry/_tracing_config.py) including:")



def search_web_tool(query: str) -> str:
    if "2006-2007" in query:
        return """Here are the total points scored by Miami Heat players in the 2006-2007 season:
        Udonis Haslem: 844 points
        Dwayne Wade: 1397 points
        James Posey: 550 points
        ...
        """
    elif "2007-2008" in query:
        return "The number of total rebounds for Dwayne Wade in the Miami Heat season 2007-2008 is 214."
    elif "2008-2009" in query:
        return "The number of total rebounds for Dwayne Wade in the Miami Heat season 2008-2009 is 398."
    return "No data found."


def percentage_change_tool(start: float, end: float) -> float:
    return ((end - start) / start) * 100


async def main() -> None:
    model_client = OllamaChatCompletionClient(model="llama3.1", request_timeout=300.0, context_window=4096)

    planning_agent = AssistantAgent(
        "PlanningAgent",
        description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
        model_client=model_client,
        system_message="""
        You are a planning agent.
        Your job is to break down complex tasks into smaller, manageable subtasks.
        Your team members are:
            WebSearchAgent: Searches for information
            DataAnalystAgent: Performs calculations

        You only plan and delegate tasks - you do not execute them yourself.

        When assigning tasks, use this format:
        1. <agent> : <task>

        After all tasks are complete, summarize the findings and end with "TERMINATE".
        """,
    )

    web_search_agent = AssistantAgent(
        "WebSearchAgent",
        description="An agent for searching information on the web.",
        tools=[search_web_tool],
        model_client=model_client,
        system_message="""
        You are a web search agent.
        Your only tool is search_tool - use it to find information.
        You make only one search call at a time.
        Once you have the results, you never do calculations based on them.
        """,
    )

    data_analyst_agent = AssistantAgent(
        "DataAnalystAgent",
        description="An agent for performing calculations.",
        model_client=model_client,
        tools=[percentage_change_tool],
        system_message="""
        You are a data analyst.
        Given the tasks you have been assigned, you should analyze the data and provide results using the tools provided.
        If you have not seen the data, ask for it.
        """,
    )

    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=25)
    termination = text_mention_termination | max_messages_termination

    selector_prompt = """Select an agent to perform task.

    {roles}

    Current conversation context:
    {history}

    Read the above conversation, then select an agent from {participants} to perform the next task.
    Make sure the planner agent has assigned tasks before other agents start working.
    Only select one agent.
    """

    task = "Who was the Miami Heat player with the highest points in the 2006-2007 season, and what was the percentage change in his total rebounds between the 2007-2008 and 2008-2009 seasons?"

    tracer = trace.get_tracer("autogen-test-agentchat")
    with tracer.start_as_current_span("runtime"):
        team = SelectorGroupChat(
            [planning_agent, web_search_agent, data_analyst_agent],
            model_client=model_client,
            termination_condition=termination,
            selector_prompt=selector_prompt,
            allow_repeated_speaker=True,
        )
        async def run_async_code_63a12af3():
            await Console(team.run_stream(task=task))
            return 
         = asyncio.run(run_async_code_63a12af3())
        logger.success(format_json())

    async def run_async_code_3902376f():
        await model_client.close()
        return 
     = asyncio.run(run_async_code_3902376f())
    logger.success(format_json())

async def run_async_code_ba09313d():
    await main()
    return 
 = asyncio.run(run_async_code_ba09313d())
logger.success(format_json())

"""
You can then use the Jaeger UI to view the traces collected from the application run above.  

![Jaeger UI](jaeger.png)



## Custom Traces 

So far, we are logging only the  default events that are generated by the AutoGen runtime (message created, publish etc). However, you can also create custom spans to log specific events in your application. 

In the example below, we will show how to log messages from the `RoundRobinGroupChat` team as they are generated by adding custom spans around the team to log runtime events and spans to log messages generated by the team.
"""
logger.info("## Custom Traces")



async def run_agents() -> None:
    model_client = OllamaChatCompletionClient(model="llama3.1", request_timeout=300.0, context_window=4096)

    primary_agent = AssistantAgent(
        "primary_agent",
        model_client=model_client,
        system_message="You are a helpful AI assistant.",
    )

    critic_agent = AssistantAgent(
        "critic_agent",
        model_client=model_client,
        system_message="Provide constructive feedback. Respond with 'APPROVE' to when your feedbacks are addressed.",
    )

    text_termination = TextMentionTermination("APPROVE")

    tracer = trace.get_tracer("autogen-test-agentchat")
    with tracer.start_as_current_span("runtime_round_robin_events"):
        team = RoundRobinGroupChat([primary_agent, critic_agent], termination_condition=text_termination)

        response_stream = team.run_stream(task="Write a 2 line haiku about the fall season")
        async for response in response_stream:
            async for response in response_stream:
                if not isinstance(response, TaskResult):
                    logger.debug(f"\n-- {response.source} -- : {response.to_text()}")
                    with tracer.start_as_current_span(f"agent_message.{response.source}") as message_span:
                        message_span.set_attribute("agent.name", response.source)
                        message_span.set_attribute("message.content", response.to_text())
                        logger.debug(f"{response.source}: {response.to_text()}")

        async def run_async_code_eecbf04a():
            await model_client.close()
            return 
         = asyncio.run(run_async_code_eecbf04a())
        logger.success(format_json())


async def run_async_code_2dce5181():
    await run_agents()
    return 
 = asyncio.run(run_async_code_2dce5181())
logger.success(format_json())

"""
In the code above, we create a new span for each message sent by the agent. We set attributes on the span to include the agent's name and the message content. This allows us to trace the flow of messages through our application and understand how they are processed.
"""
logger.info("In the code above, we create a new span for each message sent by the agent. We set attributes on the span to include the agent's name and the message content. This allows us to trace the flow of messages through our application and understand how they are processed.")

logger.info("\n\n[DONE]", bright=True)