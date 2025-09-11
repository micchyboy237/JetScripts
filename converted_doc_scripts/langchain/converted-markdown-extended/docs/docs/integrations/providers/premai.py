from jet.logger import logger
from langchain_community.chat_models import ChatPremAI
from langchain_community.embeddings import PremEmbeddings
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
# PremAI

[PremAI](https://premai.io/) is an all-in-one platform that simplifies the creation of robust, production-ready applications powered by Generative AI. By streamlining the development process, PremAI allows you to concentrate on enhancing user experience and driving overall growth for your application. You can quickly start using [our platform](https://docs.premai.io/quick-start).

## ChatPremAI

This example goes over how to use LangChain to interact with different chat models with `ChatPremAI`

### Installation and setup

We start by installing `langchain` and `premai-sdk`. You can type the following command to install:
"""
logger.info("# PremAI")

pip install premai langchain

"""
Before proceeding further, please make sure that you have made an account on PremAI and already created a project. If not, please refer to the [quick start](https://docs.premai.io/introduction) guide to get started with the PremAI platform. Create your first project and grab your API key.
"""
logger.info("Before proceeding further, please make sure that you have made an account on PremAI and already created a project. If not, please refer to the [quick start](https://docs.premai.io/introduction) guide to get started with the PremAI platform. Create your first project and grab your API key.")


"""
### Setup PremAI client in LangChain

Once we imported our required modules, let's setup our client. For now let's assume that our `project_id` is `8`. But make sure you use your project-id, otherwise it will throw error.

To use langchain with prem, you do not need to pass any model name or set any parameters with our chat-client. By default it will use the model name and parameters used in the [LaunchPad](https://docs.premai.io/get-started/launchpad).

> Note: If you change the `model` or any other parameters like `temperature`  or `max_tokens` while setting the client, it will override existing default configurations, that was used in LaunchPad.
"""
logger.info("### Setup PremAI client in LangChain")

# import getpass

if "PREMAI_API_KEY" not in os.environ:
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
You can provide system prompt here like this:
"""
logger.info("You can provide system prompt here like this:")

system_message = SystemMessage(content="You are a friendly assistant.")
human_message = HumanMessage(content="Who are you?")

chat.invoke([system_message, human_message])

"""
You can also change generation parameters while calling the model. Here's how you can do that:
"""
logger.info("You can also change generation parameters while calling the model. Here's how you can do that:")

chat.invoke(
    [system_message, human_message],
    temperature = 0.7, max_tokens = 20, top_p = 0.95
)

"""
> If you are going to place system prompt here, then it will override your system prompt that was fixed while deploying the application from the platform.

> You can find all the optional parameters [here](https://docs.premai.io/get-started/sdk#optional-parameters). Any parameters other than [these supported parameters](https://docs.premai.io/get-started/sdk#optional-parameters) will be automatically removed before calling the model.

### Native RAG Support with Prem Repositories

Prem Repositories which allows users to upload documents (.txt, .pdf etc) and connect those repositories to the LLMs. You can think Prem repositories as native RAG, where each repository can be considered as a vector database. You can connect multiple repositories. You can learn more about repositories [here](https://docs.premai.io/get-started/repositories).

Repositories are also supported in langchain premai. Here is how you can do it.
"""
logger.info("### Native RAG Support with Prem Repositories")

query = "Which models are used for dense retrieval"
repository_ids = [1985,]
repositories = dict(
    ids=repository_ids,
    similarity_threshold=0.3,
    limit=3
)

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
This is how an output looks like.
"""
logger.info("This is how an output looks like.")

Dense retrieval models typically include:

1. **BERT-based Models**: Such as DPR (Dense Passage Retrieval) which uses BERT for encoding queries and passages.
2. **ColBERT**: A model that combines BERT with late interaction mechanisms.
3. **ANCE (Approximate Nearest Neighbor Negative Contrastive Estimation)**: Uses BERT and focuses on efficient retrieval.
4. **TCT-ColBERT**: A variant of ColBERT that uses a two-tower
{
    "document_chunks": [
        {
            "repository_id": 1985,
            "document_id": 1306,
            "chunk_id": 173899,
            "document_name": "[D] Difference between sparse and dense information\u2026",
            "similarity_score": 0.3209080100059509,
            "content": "with the difference or anywhere\nwhere I can read about it?\n\n\n      17                  9\n\n\n      u/ScotiabankCanada        \u2022  Promoted\n\n\n                       Accelerate your study permit process\n                       with Scotiabank's Student GIC\n                       Program. We're here to help you tur\u2026\n\n\n                       startright.scotiabank.com         Learn More\n\n\n                            Add a Comment\n\n\nSort by:   Best\n\n\n      DinosParkour      \u2022 1y ago\n\n\n     Dense Retrieval (DR) m"
        }
    ]
}

"""
So, this also means that you do not need to make your own RAG pipeline when using the Prem Platform. Prem uses it's own RAG technology to deliver best in class performance for Retrieval Augmented Generations.

> Ideally, you do not need to connect Repository IDs here to get Retrieval Augmented Generations. You can still get the same result if you have connected the repositories in prem platform.

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
    system_prompt = "You are an helpful assistant", temperature = 0.7, max_tokens = 20
):
    sys.stdout.write(chunk.content)
    sys.stdout.flush()

"""
This will stream tokens one after the other.

> Please note: As of now, RAG with streaming is not supported. However we still support it with our API. You can learn more about that [here](https://docs.premai.io/get-started/chat-completion-sse).

## Prem Templates

Writing Prompt Templates can be super messy. Prompt templates are long, hard to manage, and must be continuously tweaked to improve and keep the same throughout the application.

With **Prem**, writing and managing prompts can be super easy. The **_Templates_** tab inside the [launchpad](https://docs.premai.io/get-started/launchpad) helps you write as many prompts you need and use it inside the SDK to make your application running using those prompts. You can read more about Prompt Templates [here](https://docs.premai.io/get-started/prem-templates).

To use Prem Templates natively with LangChain, you need to pass an id the `HumanMessage`. This id should be the name the variable of your prompt template. the `content` in `HumanMessage` should be the value of that variable.

let's say for example, if your prompt template was this:

Say hello to my name and say a feel-good quote
from my age. My name is: {name} and age is {age}

So now your human_messages should look like:
"""
logger.info("## Prem Templates")

human_messages = [
    HumanMessage(content="Shawn", id="name"),
    HumanMessage(content="22", id="age")
]

"""
Pass this `human_messages` to ChatPremAI Client. Please note: Do not forget to
pass the additional `template_id` to invoke generation with Prem Templates. If you are not aware of `template_id` you can learn more about that [in our docs](https://docs.premai.io/get-started/prem-templates). Here is an example:
"""
logger.info("Pass this `human_messages` to ChatPremAI Client. Please note: Do not forget to")

template_id = "78069ce8-xxxxx-xxxxx-xxxx-xxx"
response = chat.invoke([human_message], template_id=template_id)

"""
Prem Templates are also available for Streaming too.

## Prem Embeddings

In this section we cover how we can get access to different embedding models using `PremEmbeddings` with LangChain. Let's start by importing our modules and setting our API Key.
"""
logger.info("## Prem Embeddings")

# import getpass


if os.environ.get("PREMAI_API_KEY") is None:
#     os.environ["PREMAI_API_KEY"] = getpass.getpass("PremAI API Key:")

"""
We support lots of state of the art embedding models. You can view our list of supported LLMs and embedding models [here](https://docs.premai.io/get-started/supported-models). For now let's go for `text-embedding-3-large` model for this example. .
"""
logger.info("We support lots of state of the art embedding models. You can view our list of supported LLMs and embedding models [here](https://docs.premai.io/get-started/supported-models). For now let's go for `text-embedding-3-large` model for this example. .")

model = "text-embedding-3-large"
embedder = PremEmbeddings(project_id=8, model=model)

query = "Hello, this is a test query"
query_result = embedder.embed_query(query)


logger.debug(query_result[:5])

"""
:::note
Setting `model_name` argument in mandatory for PremAIEmbeddings unlike chat.
:::

Finally, let's embed some sample document
"""
logger.info("Setting `model_name` argument in mandatory for PremAIEmbeddings unlike chat.")

documents = [
    "This is document1",
    "This is document2",
    "This is document3"
]

doc_result = embedder.embed_documents(documents)


logger.debug(doc_result[0][:5])

"""

"""

logger.debug(f"Dimension of embeddings: {len(query_result)}")

"""
Dimension of embeddings: 3072
"""
logger.info("Dimension of embeddings: 3072")

doc_result[:5]

"""
>Result:
>
>[-0.02129288576543331,
 0.0008162345038726926,
 -0.004556538071483374,
 0.02918623760342598,
 -0.02547479420900345]

## Tool/Function Calling

LangChain PremAI supports tool/function calling. Tool/function calling allows a model to respond to a given prompt by generating output that matches a user-defined schema.

- You can learn all about tool calling in details [in our documentation here](https://docs.premai.io/get-started/function-calling).
- You can learn more about langchain tool calling in [this part of the docs](https://python.langchain.com/v0.1/docs/modules/model_io/chat/function_calling).

**NOTE:**

> The current version of LangChain ChatPremAI do not support function/tool calling with streaming support. Streaming support along with function calling will come soon.

### Passing tools to model

In order to pass tools and let the LLM choose the tool it needs to call, we need to pass a tool schema. A tool schema is the function definition along with proper docstring on what does the function do, what each argument of the function is etc. Below are some simple arithmetic functions with their schema.

**NOTE:**
> When defining function/tool schema, do not forget to add information around the function arguments, otherwise it would throw error.
"""
logger.info("## Tool/Function Calling")


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
### Binding tool schemas with our LLM

We will now use the `bind_tools` method to convert our above functions to a "tool" and binding it with the model. This means we are going to pass these tool information every time we invoke the model.
"""
logger.info("### Binding tool schemas with our LLM")

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
**Output**
"""

[{'name': 'multiply',
  'args': {'a': 3, 'b': 12},
  'id': 'call_A9FL20u12lz6TpOLaiS6rFa8'},
 {'name': 'add',
  'args': {'a': 11, 'b': 49},
  'id': 'call_MPKYGLHbf39csJIyb5BZ9xIk'}]

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
**Output**
"""

The final answers are:

- 3 * 12 = 36
- 11 + 49 = 60

"""
### Defining tool schemas: Pydantic class `Optional`

Above we have shown how to define schema using `tool` decorator, however we can equivalently define the schema using Pydantic. Pydantic is useful when your tool inputs are more complex:
"""
logger.info("### Defining tool schemas: Pydantic class `Optional`")


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
**Output**
"""

[multiply(a=3, b=12), add(a=11, b=49)]

"""
Now, as done above, we parse this and run this functions and call the LLM once again to get the result.
"""
logger.info("Now, as done above, we parse this and run this functions and call the LLM once again to get the result.")

logger.info("\n\n[DONE]", bright=True)