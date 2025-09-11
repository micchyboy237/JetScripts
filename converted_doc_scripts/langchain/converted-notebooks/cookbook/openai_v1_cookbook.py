from jet.adapters.langchain.chat_ollama import AzureOllamaEmbeddings
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.agents import AgentExecutor
from langchain.agents.ollama_assistant import OllamaAssistantRunnable
from langchain.output_parsers.ollama_tools import PydanticToolsParser
from langchain.tools import DuckDuckGoSearchRun, E2BDataAnalysisTool
from langchain.utils.ollama_functions import convert_pydantic_to_ollama_tool
from langchain_core.agents import AgentFinish
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal
import json
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
# Exploring Ollama V1 functionality

On 11.06.23 Ollama released a number of new features, and along with it bumped their Python SDK to 1.0.0. This notebook shows off the new features and how to use them with LangChain.
"""
logger.info("# Exploring Ollama V1 functionality")

# !pip install -U ollama langchain langchain-experimental


"""
## [Vision](https://platform.ollama.com/docs/guides/vision)

Ollama released multi-modal models, which can take a sequence of text and images as input.
"""
logger.info("## [Vision](https://platform.ollama.com/docs/guides/vision)")

chat = ChatOllama(model="llama3.2")
chat.invoke(
    [
        HumanMessage(
            content=[
                {"type": "text", "text": "What is this image showing"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/static/img/langchain_stack.png",
                        "detail": "auto",
                    },
                },
            ]
        )
    ]
)

"""
## [Ollama assistants](https://platform.ollama.com/docs/assistants/overview)

> The Assistants API allows you to build AI assistants within your own applications. An Assistant has instructions and can leverage models, tools, and knowledge to respond to user queries. The Assistants API currently supports three types of tools: Code Interpreter, Retrieval, and Function calling


You can interact with Ollama Assistants using Ollama tools or custom tools. When using exclusively Ollama tools, you can just invoke the assistant directly and get final answers. When using custom tools, you can run the assistant and tool execution loop using the built-in AgentExecutor or easily write your own executor.

Below we show the different ways to interact with Assistants. As a simple example, let's build a math tutor that can write and run code.

### Using only Ollama tools
"""
logger.info("## [Ollama assistants](https://platform.ollama.com/docs/assistants/overview)")


interpreter_assistant = OllamaAssistantRunnable.create_assistant(
    name="langchain assistant",
    instructions="You are a personal math tutor. Write and run code to answer math questions.",
    tools=[{"type": "code_interpreter"}],
    model="llama3.2",
)
output = interpreter_assistant.invoke({"content": "What's 10 - 4 raised to the 2.7"})
output

"""
### As a LangChain agent with arbitrary tools

Now let's recreate this functionality using our own tools. For this example we'll use the [E2B sandbox runtime tool](https://e2b.dev/docs?ref=landing-page-get-started).
"""
logger.info("### As a LangChain agent with arbitrary tools")

# !pip install e2b duckduckgo-search


tools = [E2BDataAnalysisTool(), DuckDuckGoSearchRun()]

agent = OllamaAssistantRunnable.create_assistant(
    name="langchain assistant e2b tool",
    instructions="You are a personal math tutor. Write and run code to answer math questions. You can also search the internet.",
    tools=tools,
    model="llama3.2",
    as_agent=True,
)

"""
#### Using AgentExecutor
"""
logger.info("#### Using AgentExecutor")


agent_executor = AgentExecutor(agent=agent, tools=tools)
agent_executor.invoke({"content": "What's the weather in SF today divided by 2.7"})

"""
#### Custom execution
"""
logger.info("#### Custom execution")

agent = OllamaAssistantRunnable.create_assistant(
    name="langchain assistant e2b tool",
    instructions="You are a personal math tutor. Write and run code to answer math questions.",
    tools=tools,
    model="llama3.2",
    as_agent=True,
)



def execute_agent(agent, tools, input):
    tool_map = {tool.name: tool for tool in tools}
    response = agent.invoke(input)
    while not isinstance(response, AgentFinish):
        tool_outputs = []
        for action in response:
            tool_output = tool_map[action.tool].invoke(action.tool_input)
            logger.debug(action.tool, action.tool_input, tool_output, end="\n\n")
            tool_outputs.append(
                {"output": tool_output, "tool_call_id": action.tool_call_id}
            )
        response = agent.invoke(
            {
                "tool_outputs": tool_outputs,
                "run_id": action.run_id,
                "thread_id": action.thread_id,
            }
        )

    return response

response = execute_agent(agent, tools, {"content": "What's 10 - 4 raised to the 2.7"})
logger.debug(response.return_values["output"])

next_response = execute_agent(
    agent, tools, {"content": "now add 17.241", "thread_id": response.thread_id}
)
logger.debug(next_response.return_values["output"])

"""
## [JSON mode](https://platform.ollama.com/docs/guides/text-generation/json-mode)

Constrain the model to only generate valid JSON. Note that you must include a system message with instructions to use JSON for this mode to work.

Only works with certain models.
"""
logger.info("## [JSON mode](https://platform.ollama.com/docs/guides/text-generation/json-mode)")

chat = ChatOllama(model="llama3.2").bind(
    response_format={"type": "json_object"}
)

output = chat.invoke(
    [
        SystemMessage(
            content="Extract the 'name' and 'origin' of any companies mentioned in the following statement. Return a JSON list."
        ),
        HumanMessage(
            content="Google was founded in the USA, while Deepmind was founded in the UK"
        ),
    ]
)
logger.debug(output.content)


json.loads(output.content)

"""
## [System fingerprint](https://platform.ollama.com/docs/guides/text-generation/reproducible-outputs)

Ollama sometimes changes model configurations in a way that impacts outputs. Whenever this happens, the system_fingerprint associated with a generation will change.
"""
logger.info("## [System fingerprint](https://platform.ollama.com/docs/guides/text-generation/reproducible-outputs)")

chat = ChatOllama(model="llama3.2")
output = chat.generate(
    [
        [
            SystemMessage(
                content="Extract the 'name' and 'origin' of any companies mentioned in the following statement. Return a JSON list."
            ),
            HumanMessage(
                content="Google was founded in the USA, while Deepmind was founded in the UK"
            ),
        ]
    ]
)
logger.debug(output.llm_output)

"""
## Breaking changes to Azure classes

Ollama V1 rewrote their clients and separated Azure and Ollama clients. This has led to some changes in LangChain interfaces when using Ollama V1.

BREAKING CHANGES:
- To use Azure embeddings with Ollama V1, you'll need to use the new `AzureOllamaEmbeddings` instead of the existing `OllamaEmbeddings`. `OllamaEmbeddings` continue to work when using Azure with `ollama<1`.
```python
```


RECOMMENDED CHANGES:
- When using `AzureChatOllama` or `AzureOllama`, if passing in an Azure endpoint (eg https://example-resource.azure.ollama.com/) this should be specified via the `azure_endpoint` parameter or the `AZURE_OPENAI_ENDPOINT`. We're maintaining backwards compatibility for now with specifying this via `ollama_api_base`/`base_url` or env var `OPENAI_API_BASE` but this shouldn't be relied upon.
# - When using Azure chat or embedding models, pass in API keys either via `ollama_api_key` parameter or `AZURE_OPENAI_API_KEY` parameter. We're maintaining backwards compatibility for now with specifying this via `OPENAI_API_KEY` but this shouldn't be relied upon.

## Tools

Use tools for parallel function calling.
"""
logger.info("## Breaking changes to Azure classes")




class GetCurrentWeather(BaseModel):
    """Get the current weather in a location."""

    location: str = Field(description="The city and state, e.g. San Francisco, CA")
    unit: Literal["celsius", "fahrenheit"] = Field(
        default="fahrenheit", description="The temperature unit, default to fahrenheit"
    )


prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant"), ("user", "{input}")]
)
model = ChatOllama(model="llama3.2").bind(
    tools=[convert_pydantic_to_ollama_tool(GetCurrentWeather)]
)
chain = prompt | model | PydanticToolsParser(tools=[GetCurrentWeather])

chain.invoke({"input": "what's the weather in NYC, LA, and SF"})

logger.info("\n\n[DONE]", bright=True)