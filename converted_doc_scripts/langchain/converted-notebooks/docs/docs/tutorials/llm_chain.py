from jet.file import save_file
from dotenv import load_dotenv
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
# import ChatModelTabs from "@theme/ChatModelTabs";
# import CodeBlock from "@theme/CodeBlock";
# import TabItem from '@theme/TabItem';
# import Tabs from '@theme/Tabs';
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
sidebar_position: 0
---

# Build a simple LLM application with chat models and prompt templates

In this quickstart we'll show you how to build a simple LLM application with LangChain. This application will translate text from English into another language. This is a relatively simple LLM application - it's just a single LLM call plus some prompting. Still, this is a great way to get started with LangChain - a lot of features can be built with just some prompting and an LLM call!

After reading this tutorial, you'll have a high level overview of:

- Using [language models](/docs/concepts/chat_models)

- Using [prompt templates](/docs/concepts/prompt_templates)

- Debugging and tracing your application using [LangSmith](https://docs.smith.langchain.com/)

Let's dive in!

## Setup

### Jupyter Notebook

This and other tutorials are perhaps most conveniently run in a [Jupyter notebooks](https://jupyter.org/). Going through guides in an interactive environment is a great way to better understand them. See [here](https://jupyter.org/install) for instructions on how to install.

### Installation

To install LangChain run:

<!-- HIDE_IN_NB

<Tabs>
  <TabItem value="pip" label="Pip" default>
    <CodeBlock language="bash">pip install langchain</CodeBlock>
  </TabItem>
  <TabItem value="conda" label="Conda">
    <CodeBlock language="bash">conda install langchain -c conda-forge</CodeBlock>
  </TabItem>
</Tabs>
HIDE_IN_NB -->
"""
logger.info(
    "# Build a simple LLM application with chat models and prompt templates")


"""
For more details, see our [Installation guide](/docs/how_to/installation).

### LangSmith

Many of the applications you build with LangChain will contain multiple steps with multiple invocations of LLM calls.
As these applications get more and more complex, it becomes crucial to be able to inspect what exactly is going on inside your chain or agent.
The best way to do this is with [LangSmith](https://smith.langchain.com).

After you sign up at the link above, make sure to set your environment variables to start logging traces:

```shell
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="..."
export LANGSMITH_PROJECT="default" # or any other project name
```

Or, if in a notebook, you can set them with:

```python
# import getpass

try:
    # load environment variables from .env file (requires `python-dotenv`)

    load_dotenv()
except ImportError:
    pass

os.environ["LANGSMITH_TRACING"] = "true"
if "LANGSMITH_API_KEY" not in os.environ:
#     os.environ["LANGSMITH_API_KEY"] = getpass.getpass(
        prompt="Enter your LangSmith API key (optional): "
    )
if "LANGSMITH_PROJECT" not in os.environ:
#     os.environ["LANGSMITH_PROJECT"] = getpass.getpass(
        prompt='Enter your LangSmith Project Name (default = "default"): '
    )
    if not os.environ.get("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = "default"
```

## Using Language Models

First up, let's learn how to use a language model by itself. LangChain supports many different language models that you can use interchangeably. For details on getting started with a specific model, refer to [supported integrations](/docs/integrations/chat/).

<!-- HIDE_IN_NB>

<ChatModelTabs overrideParams={{ollama: {model: "llama3.2"}}} />
HIDE_IN_NB -->
"""
logger.info("### LangSmith")


model = ChatOllama(model="llama3.2")

"""
Let's first use the model directly. [ChatModels](/docs/concepts/chat_models) are instances of LangChain [Runnables](/docs/concepts/runnables/), which means they expose a standard interface for interacting with them. To simply call the model, we can pass in a list of [messages](/docs/concepts/messages/) to the `.invoke` method.
"""
logger.info("Let's first use the model directly. [ChatModels](/docs/concepts/chat_models) are instances of LangChain [Runnables](/docs/concepts/runnables/), which means they expose a standard interface for interacting with them. To simply call the model, we can pass in a list of [messages](/docs/concepts/messages/) to the `.invoke` method.")


messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi!"),
]


response = model.invoke(messages)
save_file(response, f"{OUTPUT_DIR}/response.json")

"""
:::tip

If we've enabled LangSmith, we can see that this run is logged to LangSmith, and can see the [LangSmith trace](https://docs.smith.langchain.com/observability/concepts#traces). The LangSmith trace reports [token](/docs/concepts/tokens/) usage information, latency, [standard model parameters](/docs/concepts/chat_models/#standard-parameters) (such as temperature), and other information.

:::

Note that ChatModels receive [message](/docs/concepts/messages/) objects as input and generate message objects as output. In addition to text content, message objects convey conversational [roles](/docs/concepts/messages/#role) and hold important data, such as [tool calls](/docs/concepts/tool_calling/) and token usage counts.

LangChain also supports chat model inputs via strings or [Ollama format](/docs/concepts/messages/#ollama-format). The following are equivalent:

```python
model.invoke("Hello")

model.invoke([{"role": "user", "content": "Hello"}])

model.invoke([HumanMessage("Hello")])
```

### Streaming

Because chat models are [Runnables](/docs/concepts/runnables/), they expose a standard interface that includes async and streaming modes of invocation. This allows us to stream individual tokens from a chat model:
"""
logger.info("### Streaming")

stream_response = ""
content = ""
for token in model.stream(messages):
    logger.debug(token.content, end="|")
    content += token.content
    stream_response = token
stream_response.content = content
save_file(stream_response, f"{OUTPUT_DIR}/stream_response.md")

"""
You can find more details on streaming chat model outputs in [this guide](/docs/how_to/chat_streaming/).

## Prompt Templates

Right now we are passing a list of messages directly into the language model. Where does this list of messages come from? Usually, it is constructed from a combination of user input and application logic. This application logic usually takes the raw user input and transforms it into a list of messages ready to pass to the language model. Common transformations include adding a system message or formatting a template with the user input.

[Prompt templates](/docs/concepts/prompt_templates/) are a concept in LangChain designed to assist with this transformation. They take in raw user input and return data (a prompt) that is ready to pass into a language model. 

Let's create a prompt template here. It will take in two user variables:

- `language`: The language to translate text into
- `text`: The text to translate
"""
logger.info("## Prompt Templates")


system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

"""
Note that `ChatPromptTemplate` supports multiple [message roles](/docs/concepts/messages/#role) in a single template. We format the `language` parameter into the system message, and the user `text` into a user message.

The input to this prompt template is a dictionary. We can play around with this prompt template by itself to see what it does by itself
"""
logger.info(
    "Note that `ChatPromptTemplate` supports multiple [message roles](/docs/concepts/messages/#role) in a single template. We format the `language` parameter into the system message, and the user `text` into a user message.")

prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})

prompt

"""
We can see that it returns a `ChatPromptValue` that consists of two messages. If we want to access the messages directly we do:
"""
logger.info("We can see that it returns a `ChatPromptValue` that consists of two messages. If we want to access the messages directly we do:")

prompt.to_messages()

"""
Finally, we can invoke the chat model on the formatted prompt:
"""
logger.info("Finally, we can invoke the chat model on the formatted prompt:")

response = model.invoke(prompt)
save_file(response, f"{OUTPUT_DIR}/formatted_prompt_response.json")

"""
:::tip
Message `content` can contain both text and [content blocks](/docs/concepts/messages/#aimessage) with additional structure. See [this guide](/docs/how_to/output_parser_string/) for more information.
:::

If we take a look at the [LangSmith trace](https://smith.langchain.com/public/3ccc2d5e-2869-467b-95d6-33a577df99a2/r), we can see exactly what prompt the chat model receives, along with [token](/docs/concepts/tokens/) usage information, latency, [standard model parameters](/docs/concepts/chat_models/#standard-parameters) (such as temperature), and other information.

## Conclusion

That's it! In this tutorial you've learned how to create your first simple LLM application. You've learned how to work with language models, how to create a prompt template, and how to get great observability into applications you create with LangSmith.

This just scratches the surface of what you will want to learn to become a proficient AI Engineer. Luckily - we've got a lot of other resources!

For further reading on the core concepts of LangChain, we've got detailed [Conceptual Guides](/docs/concepts).

If you have more specific questions on these concepts, check out the following sections of the how-to guides:

- [Chat models](/docs/how_to/#chat-models)
- [Prompt templates](/docs/how_to/#prompt-templates)

And the LangSmith docs:

- [LangSmith](https://docs.smith.langchain.com)
"""
logger.info("## Conclusion")

logger.info("\n\n[DONE]", bright=True)
