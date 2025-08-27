import asyncio
from jet.transformers.formatters import format_json
from azure.identity import AzureCliCredential
from jet.logger import CustomLogger
from samples.service_settings import ServiceSettings
from semantic_kernel import Kernel
from semantic_kernel import __version__
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings, OllamaChatPromptExecutionSettings
from semantic_kernel.connectors.ai.open_ai import OllamaChatCompletion
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.prompt_template.input_variable import InputVariable
from services import Service
import os
import shutil
import sys


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Creating a basic chat experience with kernel arguments

In this example, we show how you can build a simple chat bot by sending and updating the kernel arguments with your requests.

We introduce the Kernel Arguments object which in this demo functions similarly as a key-value store that you can use when running the kernel.

The chat history is local (i.e. in your computer's RAM) and not persisted anywhere beyond the life of this Jupyter session.

In future examples, we will show how to persist the chat history on disk so that you can bring it into your applications.

In this chat scenario, as the user talks back and forth with the bot, the chat context gets populated with the history of the conversation. During each new run of the kernel, the kernel arguments and chat history can provide the AI with its variables' content.
"""
logger.info("# Creating a basic chat experience with kernel arguments")

# %pip install -U semantic-kernel

__version__

"""
Initial configuration for the notebook to run properly.
"""
logger.info("Initial configuration for the notebook to run properly.")


notebook_dir = os.path.abspath("")
parent_dir = os.path.dirname(notebook_dir)
grandparent_dir = os.path.dirname(parent_dir)


sys.path.append(grandparent_dir)

"""
### Configuring the Kernel

Let's get started with the necessary configuration to run Semantic Kernel. For Notebooks, we require a `.env` file with the proper settings for the model you use. Create a new file named `.env` and place it in this directory. Copy the contents of the `.env.example` file from this directory and paste it into the `.env` file that you just created.

**NOTE: Please make sure to include `GLOBAL_LLM_SERVICE` set to either Ollama, AzureOllama, or HuggingFace in your .env file. If this setting is not included, the Service will default to AzureOllama.**

#### Option 1: using Ollama

Add your [Ollama Key](https://openai.com/product/) key to your `.env` file (org Id only if you have multiple orgs):

```
GLOBAL_LLM_SERVICE="Ollama"
# OPENAI_API_KEY="sk-..."
OPENAI_ORG_ID=""
OPENAI_CHAT_MODEL_ID=""
OPENAI_TEXT_MODEL_ID=""
OPENAI_EMBEDDING_MODEL_ID=""
```
The names should match the names used in the `.env` file, as shown above.

#### Option 2: using Azure Ollama

Add your [Azure Open AI Service key](https://learn.microsoft.com/azure/cognitive-services/openai/quickstart?pivots=programming-language-studio) settings to the `.env` file in the same folder:

```
GLOBAL_LLM_SERVICE="AzureOllama"
# AZURE_OPENAI_API_KEY="..."
AZURE_OPENAI_ENDPOINT="https://..."
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="..."
AZURE_OPENAI_TEXT_DEPLOYMENT_NAME="..."
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME="..."
AZURE_OPENAI_API_VERSION="..."
```
The names should match the names used in the `.env` file, as shown above.

# As alternative to `AZURE_OPENAI_API_KEY`, it's possible to authenticate using `credential` parameter, more information here: [Azure Identity](https://learn.microsoft.com/en-us/python/api/overview/azure/identity-readme).

In the following example, `AzureCliCredential` is used. To authenticate using Azure CLI:

1. Install [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli).
2. Run `az login` command in terminal and follow the authentication steps.


For more advanced configuration, please follow the steps outlined in the [setup guide](./CONFIGURING_THE_KERNEL.md).

We will load our settings and get the LLM service to use for the notebook.
"""
logger.info("### Configuring the Kernel")



service_settings = ServiceSettings()

selectedService = (
    Service.AzureOllama
    if service_settings.global_llm_service is None
    else Service(service_settings.global_llm_service.lower())
)
logger.debug(f"Using service type: {selectedService}")


kernel = Kernel()

service_id = None
if selectedService == Service.Ollama:

    service_id = "default"
    kernel.add_service(
        OllamaChatCompletion(
            service_id=service_id,
        ),
    )
elif selectedService == Service.AzureOllama:


    service_id = "default"
    kernel.add_service(
        AzureChatCompletion(service_id=service_id, credential=AzureCliCredential()),
    )

"""
Let's define a prompt outlining a dialogue chat bot.
"""
logger.info("Let's define a prompt outlining a dialogue chat bot.")

prompt = """
ChatBot can have a conversation with you about any topic.
It can give explicit instructions or say 'I don't know' if it does not have an answer.

{{$history}}
User: {{$user_input}}
ChatBot: """

"""
Register your semantic function
"""
logger.info("Register your semantic function")


if selectedService == Service.Ollama:
    execution_settings = OllamaChatPromptExecutionSettings(
        service_id=service_id,
        ai_model_id="gpt-3.5-turbo",
        max_tokens=2000,
        temperature=0.7,
    )
elif selectedService == Service.AzureOllama:
    execution_settings = AzureChatPromptExecutionSettings(
        service_id=service_id,
        ai_model_id="gpt-35-turbo",
        max_tokens=2000,
        temperature=0.7,
    )

prompt_template_config = PromptTemplateConfig(
    template=prompt,
    name="chat",
    template_format="semantic-kernel",
    input_variables=[
        InputVariable(name="user_input", description="The user input", is_required=True),
        InputVariable(
            name="history", description="The conversation history", is_required=True, allow_dangerously_set_content=True
        ),
    ],
    execution_settings=execution_settings,
)

chat_function = kernel.add_function(
    function_name="chat",
    plugin_name="chatPlugin",
    prompt_template_config=prompt_template_config,
)


chat_history = ChatHistory()
chat_history.add_system_message("You are a helpful chatbot who is good about giving book recommendations.")

"""
Initialize the Kernel Arguments
"""
logger.info("Initialize the Kernel Arguments")


arguments = KernelArguments(user_input="Hi, I'm looking for book suggestions", history=chat_history)

"""
Chat with the Bot
"""
logger.info("Chat with the Bot")

async def run_async_code_f0ff12b2():
    async def run_async_code_65c53b37():
        response = await kernel.invoke(chat_function, arguments)
        return response
    response = asyncio.run(run_async_code_65c53b37())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_f0ff12b2())
logger.success(format_json(response))
logger.debug(response)

"""
Update the history with the output
"""
logger.info("Update the history with the output")

chat_history.add_assistant_message(str(response))

"""
Keep Chatting!
"""
logger.info("Keep Chatting!")

async def chat(input_text: str) -> None:
    logger.debug(f"User: {input_text}")

    async def run_async_code_2f291ec1():
        async def run_async_code_8d9feb1d():
            answer = await kernel.invoke(chat_function, KernelArguments(user_input=input_text, history=chat_history))
            return answer
        answer = asyncio.run(run_async_code_8d9feb1d())
        logger.success(format_json(answer))
        return answer
    answer = asyncio.run(run_async_code_2f291ec1())
    logger.success(format_json(answer))

    logger.debug(f"ChatBot: {answer}")

    chat_history.add_user_message(input_text)
    chat_history.add_assistant_message(str(answer))

async def run_async_code_e6e647b0():
    await chat("I love history and philosophy, I'd like to learn something new about Greece, any suggestion?")
    return 
 = asyncio.run(run_async_code_e6e647b0())
logger.success(format_json())

async def run_async_code_a89b12f2():
    await chat("that sounds interesting, what is it about?")
    return 
 = asyncio.run(run_async_code_a89b12f2())
logger.success(format_json())

async def run_async_code_01aca0fb():
    await chat("if I read that book, what exactly will I learn about Greek history?")
    return 
 = asyncio.run(run_async_code_01aca0fb())
logger.success(format_json())

async def run_async_code_bb0f9561():
    await chat("could you list some more books I could read about this topic?")
    return 
 = asyncio.run(run_async_code_bb0f9561())
logger.success(format_json())

"""
After chatting for a while, we have built a growing history, which we are attaching to each prompt and which contains the full conversation. Let's take a look!
"""
logger.info("After chatting for a while, we have built a growing history, which we are attaching to each prompt and which contains the full conversation. Let's take a look!")

logger.debug(chat_history)

logger.info("\n\n[DONE]", bright=True)