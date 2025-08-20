from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llms import ChatMessage
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.premai import PremAI
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
<a href="https://colab.research.google.com/drive/1F0V_eClPOpS_2HIW-F2mCivgqBfFo2TR?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# PremAI LlamaIndex

[PremAI](https://premai.io/) is an all-in-one platform that simplifies the creation of robust, production-ready applications powered by Generative AI. By streamlining the development process, PremAI allows you to concentrate on enhancing user experience and driving overall growth for your application. You can quickly start using our platform [here](https://docs.premai.io/quick-start).

## Installation and setup

We start by installing `llama-index` and `premai-sdk`. You can type the following command to install:

```bash
pip install premai llama-index
```

Before proceeding further, please make sure that you have made an account on PremAI and already created a project. If not, please refer to the [quick start](https://docs.premai.io/introduction) guide to get started with the PremAI platform. Create your first project and grab your API key.
"""
logger.info("# PremAI LlamaIndex")

# %pip install llama-index-llms-premai


"""
## Setup PremAI client with LlamaIndex

Once we imported our required modules, let's setup our client. For now let's assume that our `project_id` is `8`. But make sure you use your project-id, otherwise it will throw error.

In order to use llama-index with PremAI, you do not need to pass any model name or set any parameters with our chat-client. By default it will use the model name and parameters used in the [LaunchPad](https://docs.premai.io/get-started/launchpad).

> If you change the `model` or any other parameters like `temperature`  or `max_tokens` while setting the client, it will override existing default configurations, that was used in LaunchPad.
"""
logger.info("## Setup PremAI client with LlamaIndex")

# import getpass

if os.environ.get("PREMAI_API_KEY") is None:
#     os.environ["PREMAI_API_KEY"] = getpass.getpass("PremAI API Key:")

prem_chat = PremAI(project_id=8)

"""
## Chat Completions

Now you are all set. We can now start with interacting with our application. Let's start by building simple chat request and responses using llama-index.
"""
logger.info("## Chat Completions")

messages = [
    ChatMessage(role="user", content="What is your name"),
    ChatMessage(
        role="user", content="Write an essay about your school in 500 words"
    ),
]

"""
Please note that: You can provide system prompt in `ChatMessage` like this:

```python
messages = [
    ChatMessage(role="system", content="Act like a pirate"),
    ChatMessage(role="user", content="What is your name"),
    ChatMessage(role="user", content="Where do you live, write an essay in 500 words"),
]
```
Additionally, you can also instantiate your client with system prompt like this:

```python
chat = PremAI(project_id=8, system_prompt="Act like nemo fish")
```


> In both of the scenarios, you are going to override your system prompt that was fixed while deploying the application from the platform. And, specifically in this case, if you override system prompt while instantiating the **PremAI** class then system message in `ChatMessage` won't provide any affect.

> So if you want to override system prompt for any experimental cases, either you need to provide that while instantiating the client or while writing it in `ChatMessage` with a role `system`.


Now let's call the model
"""
logger.info("Please note that: You can provide system prompt in `ChatMessage` like this:")

response = prem_chat.chat(messages)
logger.debug(response)

"""
You can also convert your chat function to a completion function. Here's how it works
"""
logger.info("You can also convert your chat function to a completion function. Here's how it works")

completion = prem_chat.complete("Paul Graham is ")

"""
> If you are going to place system prompt here, then it will override your system prompt that was fixed while deploying the application from the platform. 

> You can find all the optional parameters [here](https://docs.premai.io/get-started/sdk#optional-parameters). Any parameters other than [these supported parameters](https://docs.premai.io/get-started/sdk#optional-parameters) will be automatically removed before calling the model.

## Native RAG Support with Prem Repositories

Prem Repositories which allows users to upload documents (.txt, .pdf etc) and connect those repositories to the LLMs. You can think Prem repositories as native RAG, where each repository can be considered as a vector database. You can connect multiple repositories. You can learn more about repositories [here](https://docs.premai.io/get-started/repositories).

Repositories are also supported in langchain premai. Here is how you can do it.
"""
logger.info("## Native RAG Support with Prem Repositories")

query = "what is the diameter of individual Galaxy"
repository_ids = [
    1991,
]
repositories = dict(ids=repository_ids, similarity_threshold=0.3, limit=3)

"""
First we start by defining our repository with some repository ids. Make sure that the ids are valid repository ids. You can learn more about how to get the repository id [here](https://docs.premai.io/get-started/repositories). 

> Please note: Similar like `model_name` when you invoke the argument `repositories`, then you are potentially overriding the repositories connected in the launchpad. 

Now, we connect the repository with our chat object to invoke RAG based generations.
"""
logger.info("First we start by defining our repository with some repository ids. Make sure that the ids are valid repository ids. You can learn more about how to get the repository id [here](https://docs.premai.io/get-started/repositories).")

messages = [
    ChatMessage(role="user", content=query),
]

response = prem_chat.chat(messages, repositories=repositories)
logger.debug(response)

"""
So, this also means that you do not need to make your own RAG pipeline when using the Prem Platform. Prem uses it's own RAG technology to deliver best in class performance for Retrieval Augmented Generations.

> Ideally, you do not need to connect Repository IDs here to get Retrieval Augmented Generations. You can still get the same result if you have connected the repositories in prem platform.

## Prem Templates 

Writing Prompt Templates can be super messy. Prompt templates are long, hard to manage, and must be continuously tweaked to improve and keep the same throughout the application. 

With **Prem**, writing and managing prompts can be super easy. The **_Templates_** tab inside the [launchpad](https://docs.premai.io/get-started/launchpad) helps you write as many prompts templates you need and use it inside the SDK to make your application running using those prompts. You can read more about Prompt Templates [here](https://docs.premai.io/get-started/prem-templates). 

To use Prem Templates natively with llama-index, you need to pass a dictionary which will contain an `id` as key and the value will be the template variable. This key-value should be passed in the `additional_kwargs` of ChatMessage. 

let's say for example, if your prompt template was this:

```text
Say hello to my name and say a feel-good quote
from my age. My name is: {name} and age is {age}
```

So now your message should look like:
"""
logger.info("## Prem Templates")

messages = [
    ChatMessage(
        role="user", content="Shawn", additional_kwargs={"id": "name"}
    ),
    ChatMessage(role="user", content="22", additional_kwargs={"id": "age"}),
]

"""
Pass this `messages` to llama-index PremAI Client. Please note: Do not forget to
pass the additional `template_id` to invoke generation with Prem Templates. If you are not aware of `template_id` you can learn more about that [in our docs](https://docs.premai.io/get-started/prem-templates). Here is an example:
"""
logger.info("Pass this `messages` to llama-index PremAI Client. Please note: Do not forget to")

template_id = "78069ce8-xxxxx-xxxxx-xxxx-xxx"
response = prem_chat.chat(messages, template_id=template_id)

"""
Prem Templates are also available for Streaming too.

## Streaming 

In this section, let's see how we can stream tokens using llama-index and PremAI. It is very similar to above methods. Here's how you do it.
"""
logger.info("## Streaming")

streamed_response = prem_chat.stream_chat(messages)

for response_delta in streamed_response:
    logger.debug(response_delta.delta, end="")

"""
This will stream tokens one after the other. Similar to `complete` method, we have `stream_complete` method which does streaming of tokens for completion.
"""
logger.info("This will stream tokens one after the other. Similar to `complete` method, we have `stream_complete` method which does streaming of tokens for completion.")

streamed_response = prem_chat.stream_complete("hello how are you")

for response_delta in streamed_response:
    logger.debug(response_delta.delta, end="")

logger.info("\n\n[DONE]", bright=True)