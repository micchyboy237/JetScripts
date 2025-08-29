from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
from llama_index.core.agent import ReActAgent
from jet.llm.ollama.base import Ollama
from llama_index.core.llms import ChatMessage
from jet.llm.tools.function_tool import FunctionTool
from llama_index.core import PromptTemplate
from llama_index.core.agent import ReActChatFormatter
from llama_index.core.llms import MessageRole

initialize_ollama_settings()

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/agent/react_agent.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

"""
# ReAct Agent - A Simple Intro with Calculator Tools

This is a notebook that showcases the ReAct agent over very simple calculator tools (no fancy RAG pipelines or API calls).

We show how it can reason step-by-step over different tools to achieve the end goal.
"""

"""
If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""

# %pip install llama-index-llms-ollama

# !pip install llama-index


"""
## Define Function Tools

We setup some trivial `multiply` and `add` tools. Note that you can define arbitrary functions and pass it to the `FunctionTool` (which will process the docstring and parameter signature).
"""


def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


add_tool = FunctionTool.from_defaults(fn=add)

"""
## Run Some Queries

### gpt-3.5-turbo
"""

llm = Ollama(model="llama3.1")
agent = ReActAgent.from_tools([multiply_tool, add_tool], llm=llm, verbose=True)

response = agent.chat("What is 20+(2*4)? Calculate step by step ")

response_gen = agent.stream_chat("What is 20+2*4? Calculate step by step")
response_gen.print_response_stream()

"""
### gpt-4
"""

llm = Ollama(model="llama3.1")
agent = ReActAgent.from_tools([multiply_tool, add_tool], llm=llm, verbose=True)

response = agent.chat("What is 2+2*4")
print(response)

"""
## View Prompts

Let's take a look at the core system prompt powering the ReAct agent! 

Within the agent, the current conversation history is dumped below this line.
"""

llm = Ollama(model="llama3.1")
agent = ReActAgent.from_tools([multiply_tool, add_tool], llm=llm, verbose=True)

prompt_dict = agent.get_prompts()
for k, v in prompt_dict.items():
    print(f"Prompt: {k}\n\nValue: {v.template}")

"""
### Customizing the Prompt

For fun, let's try instructing the agent to output the answer along with reasoning in bullet points. See "## Additional Rules" section.
"""


react_system_header_str = """\

You are designed to help with a variety of tasks, from answering questions \
    to providing summaries to other types of analyses.

You have access to a wide variety of tools. You are responsible for using
the tools in any sequence you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools
to complete each subtask.

You have access to the following tools:
{tool_desc}

To answer the question, please use the following format.

```
Thought: I need to use a tool to help me answer the question.
Action: tool name (one of {tool_names}) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
```

Please ALWAYS start with a Thought.

Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.

If this format is used, the user will respond in the following format:

```
Observation: tool response
```

You should keep repeating the above format until you have enough information
to answer the question without using any more tools. At that point, you MUST respond
in the one of the following two formats:

```
Thought: I can answer without using any more tools.
Answer: [your answer here]
```

```
Thought: I cannot answer the question with the provided tools.
Answer: Sorry, I cannot answer your query.
```

- The answer MUST contain a sequence of bullet points that explain how you arrived at the answer. This can include aspects of the previous conversation history.
- You MUST obey the function signature of each tool. Do NOT pass in no arguments if the function expects arguments.

Below is the current conversation consisting of interleaving human and assistant messages.

"""
react_system_prompt = PromptTemplate(react_system_header_str)

agent.get_prompts()

agent.update_prompts({"agent_worker:system_prompt": react_system_prompt})

agent.reset()
response = agent.chat("What is 5+3+2")
print(response)

"""
### Customizing the Message Role of Observation

If the LLM you use supports function/tool calling, you may set the message role of observations to `MessageRole.TOOL`.  
Doing this will prevent the tool outputs from being misinterpreted as new user messages for some models.
"""


agent = ReActAgent.from_tools(
    [multiply_tool, add_tool],
    llm=llm,
    react_chat_formatter=ReActChatFormatter.from_defaults(
        observation_role=MessageRole.TOOL
    ),
    verbose=True,
)

logger.info("\n\n[DONE]", bright=True)
