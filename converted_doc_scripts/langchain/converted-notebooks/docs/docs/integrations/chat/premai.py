from jet.logger import logger
from langchain_community.chat_models import ChatPremAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages import ToolMessage
from langchain_core.output_parsers.ollama_tools import PydanticToolsParser
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import json
import os
import shutil
import sys


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
sidebar_label: PremAI
---

# ChatPremAI

[PremAI](https://premai.io/) is an all-in-one platform that simplifies the creation of robust, production-ready applications powered by Generative AI. By streamlining the development process, PremAI allows you to concentrate on enhancing user experience and driving overall growth for your application. You can quickly start using our platform [here](https://docs.premai.io/quick-start).

This example goes over how to use LangChain to interact with different chat models with `ChatPremAI`

### Installation and setup

We start by installing `langchain` and `premai-sdk`. You can type the following command to install:

```bash
pip install premai langchain
```

Before proceeding further, please make sure that you have made an account on PremAI and already created a project. If not, please refer to the [quick start](https://docs.premai.io/introduction) guide to get started with the PremAI platform. Create your first project and grab your API key.
"""
logger.info("# ChatPremAI")


"""
### Setup PremAI client in LangChain

Once we imported our required modules, let's setup our client. For now let's assume that our `project_id` is `8`. But make sure you use your project-id, otherwise it will throw error.

To use langchain with prem, you do not need to pass any model name or set any parameters with our chat-client. By default it will use the model name and parameters used in the [LaunchPad](https://docs.premai.io/get-started/launchpad). 

> Note: If you change the `model` or any other parameters like `temperature`  or `max_tokens` while setting the client, it will override existing default configurations, that was used in LaunchPad.
"""
logger.info("### Setup PremAI client in LangChain")

# import getpass


if os.environ.get("PREMAI_API_KEY") is None:
#     os.environ["PREMAI_API_KEY"] = getpass.getpass("PremAI API Key:")

chat = ChatPremAI(project_id=1234, model_name="gpt-4o")

"""
### Chat Completions

`ChatPremAI` supports two methods: `invoke` (which is the same as `generate`) and `stream`. 

The first one will give us a static result. Whereas the second one will stream tokens one by one. Here's how you can generate chat-like completions.
"""
logger.info("### Chat Completions")

human_message = HumanMessage(content="Who are you?")

response = chat.invoke([human_message])
logger.debug(response.content)

"""
Above looks interesting right? I set my default lanchpad system-prompt as: `Always sound like a pirate` You can also, override the default system prompt if you need to. Here's how you can do it.
"""
logger.info("Above looks interesting right? I set my default lanchpad system-prompt as: `Always sound like a pirate` You can also, override the default system prompt if you need to. Here's how you can do it.")

system_message = SystemMessage(content="You are a friendly assistant.")
human_message = HumanMessage(content="Who are you?")

chat.invoke([system_message, human_message])

"""
You can provide system prompt here like this:
"""
logger.info("You can provide system prompt here like this:")

chat.invoke([system_message, human_message], temperature=0.7, max_tokens=10, top_p=0.95)

"""
> If you are going to place system prompt here, then it will override your system prompt that was fixed while deploying the application from the platform.

### Native RAG Support with Prem Repositories

Prem Repositories which allows users to upload documents (.txt, .pdf etc) and connect those repositories to the LLMs. You can think Prem repositories as native RAG, where each repository can be considered as a vector database. You can connect multiple repositories. You can learn more about repositories [here](https://docs.premai.io/get-started/repositories).

Repositories are also supported in langchain premai. Here is how you can do it.
"""
logger.info("### Native RAG Support with Prem Repositories")

query = "Which models are used for dense retrieval"
repository_ids = [
    1985,
]
repositories = dict(ids=repository_ids, similarity_threshold=0.3, limit=3)

"""
First we start by defining our repository with some repository ids. Make sure that the ids are valid repository ids. You can learn more about how to get the repository id [here](https://docs.premai.io/get-started/repositories). 

> Please note: Similar like `model_name` when you invoke the argument `repositories`, then you are potentially overriding the repositories connected in the launchpad. 

Now, we connect the repository with our chat object to invoke RAG based generations.
"""
logger.info("First we start by defining our repository with some repository ids. Make sure that the ids are valid repository ids. You can learn more about how to get the repository id [here](https://docs.premai.io/get-started/repositories).")


response = chat.invoke(query, max_tokens=100, repositories=repositories)

logger.debug(response.content)
logger.debug(json.dumps(response.response_metadata, indent=4))

"""
> Ideally, you do not need to connect Repository IDs here to get Retrieval Augmented Generations. You can still get the same result if you have connected the repositories in prem platform.

### Prem Templates

Writing Prompt Templates can be super messy. Prompt templates are long, hard to manage, and must be continuously tweaked to improve and keep the same throughout the application. 

With **Prem**, writing and managing prompts can be super easy. The **_Templates_** tab inside the [launchpad](https://docs.premai.io/get-started/launchpad) helps you write as many prompts you need and use it inside the SDK to make your application running using those prompts. You can read more about Prompt Templates [here](https://docs.premai.io/get-started/prem-templates). 

To use Prem Templates natively with LangChain, you need to pass an id the `HumanMessage`. This id should be the name the variable of your prompt template. the `content` in `HumanMessage` should be the value of that variable. 

let's say for example, if your prompt template was this:

```text
Say hello to my name and say a feel-good quote
from my age. My name is: {name} and age is {age}
```

So now your human_messages should look like:
"""
logger.info("### Prem Templates")

human_messages = [
    HumanMessage(content="Shawn", id="name"),
    HumanMessage(content="22", id="age"),
]

"""
Pass this `human_messages` to ChatPremAI Client. Please note: Do not forget to
pass the additional `template_id` to invoke generation with Prem Templates. If you are not aware of `template_id` you can learn more about that [in our docs](https://docs.premai.io/get-started/prem-templates). Here is an example:
"""
logger.info("Pass this `human_messages` to ChatPremAI Client. Please note: Do not forget to")

template_id = "78069ce8-xxxxx-xxxxx-xxxx-xxx"
response = chat.invoke([human_messages], template_id=template_id)
logger.debug(response.content)

"""
Prem Template feature is available in streaming too.

### Streaming

In this section, let's see how we can stream tokens using langchain and PremAI. Here's how you do it.
"""
logger.info("### Streaming")


for chunk in chat.stream("hello how are you"):
    sys.stdout.write(chunk.content)
    sys.stdout.flush()

"""
Similar to above, if you want to override the system-prompt and the generation parameters, you need to add the following:
"""
logger.info("Similar to above, if you want to override the system-prompt and the generation parameters, you need to add the following:")



for chunk in chat.stream(
    "hello how are you",
    system_prompt="act like a dog",
    temperature=0.7,
    max_tokens=200,
):
    sys.stdout.write(chunk.content)
    sys.stdout.flush()

"""
### Tool/Function Calling

LangChain PremAI supports tool/function calling. Tool/function calling allows a model to respond to a given prompt by generating output that matches a user-defined schema. 

- You can learn all about tool calling in details [in our documentation here](https://docs.premai.io/get-started/function-calling).
- You can learn more about langchain tool calling in [this part of the docs](https://python.langchain.com/v0.1/docs/modules/model_io/chat/function_calling).

**NOTE:**
The current version of LangChain ChatPremAI do not support function/tool calling with streaming support. Streaming support along with function calling will come soon. 

#### Passing tools to model

In order to pass tools and let the LLM choose the tool it needs to call, we need to pass a tool schema. A tool schema is the function definition along with proper docstring on what does the function do, what each argument of the function is etc. Below are some simple arithmetic functions with their schema. 

**NOTE:** When defining function/tool schema, do not forget to add information around the function arguments, otherwise it would throw error.
"""
logger.info("### Tool/Function Calling")



class OperationInput(BaseModel):
    a: int = Field(description="First number")
    b: int = Field(description="Second number")


@tool("add", args_schema=OperationInput, return_direct=True)
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


@tool("multiply", args_schema=OperationInput, return_direct=True)
def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

"""
#### Binding tool schemas with our LLM

We will now use the `bind_tools` method to convert our above functions to a "tool" and binding it with the model. This means we are going to pass these tool informations everytime we invoke the model.
"""
logger.info("#### Binding tool schemas with our LLM")

tools = [add, multiply]
llm_with_tools = chat.bind_tools(tools)

"""
After this, we get the response from the model which is now binded with the tools.
"""
logger.info("After this, we get the response from the model which is now binded with the tools.")

query = "What is 3 * 12? Also, what is 11 + 49?"

messages = [HumanMessage(query)]
ai_msg = llm_with_tools.invoke(messages)

"""
As we can see, when our chat model is binded with tools, then based on the given prompt, it calls the correct set of the tools and sequentially.
"""
logger.info("As we can see, when our chat model is binded with tools, then based on the given prompt, it calls the correct set of the tools and sequentially.")

ai_msg.tool_calls

"""
We append this message shown above to the LLM which acts as a context and makes the LLM aware that what all functions it has called.
"""
logger.info("We append this message shown above to the LLM which acts as a context and makes the LLM aware that what all functions it has called.")

messages.append(ai_msg)

"""
Since tool calling happens into two phases, where:

1. in our first call, we gathered all the tools that the LLM decided to tool, so that it can get the result as an added context to give more accurate and hallucination free result. 

2. in our second call, we will parse those set of tools decided by LLM and run them (in our case it will be the functions we defined, with the LLM's extracted arguments) and pass this result to the LLM
"""
logger.info("Since tool calling happens into two phases, where:")


for tool_call in ai_msg.tool_calls:
    selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
    tool_output = selected_tool.invoke(tool_call["args"])
    messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

"""
Finally, we call the LLM (binded with the tools) with the function response added in it's context.
"""
logger.info("Finally, we call the LLM (binded with the tools) with the function response added in it's context.")

response = llm_with_tools.invoke(messages)
logger.debug(response.content)

"""
### Defining tool schemas: Pydantic class

Above we have shown how to define schema using `tool` decorator, however we can equivalently define the schema using Pydantic. Pydantic is useful when your tool inputs are more complex:
"""
logger.info("### Defining tool schemas: Pydantic class")



class add(BaseModel):
    """Add two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


class multiply(BaseModel):
    """Multiply two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


tools = [add, multiply]

"""
Now, we can bind them to chat models and directly get the result:
"""
logger.info("Now, we can bind them to chat models and directly get the result:")

chain = llm_with_tools | PydanticToolsParser(tools=[multiply, add])
chain.invoke(query)

"""
Now, as done above, we parse this and run this functions and call the LLM once again to get the result.
"""
logger.info("Now, as done above, we parse this and run this functions and call the LLM once again to get the result.")

logger.info("\n\n[DONE]", bright=True)