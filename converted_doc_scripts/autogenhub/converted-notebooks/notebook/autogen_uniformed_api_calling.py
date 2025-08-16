from autogen import OllamaWrapper
from jet.logger import CustomLogger
import autogen
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# A Uniform interface to call different LLMs

Autogen provides a uniform interface for API calls to different LLMs, and creating LLM agents from them.
Through setting up a configuration file, you can easily switch between different LLMs by just changing the model name, while enjoying all the [enhanced features](https://autogenhub.github.io/autogen/docs/topics/llm-caching) such as [caching](https://autogenhub.github.io/autogen/docs/Use-Cases/enhanced_inference/#usage-summary) and [cost calculation](https://autogenhub.github.io/autogen/docs/Use-Cases/enhanced_inference/#usage-summary)!

In this notebook, we will show you how to use AutoGen to call different LLMs and create LLM agents from them.

Currently, we support the following model families:
- [Ollama](https://platform.openai.com/docs/overview)
- [Azure Ollama](https://azure.microsoft.com/en-us/products/ai-services/openai-service/?ef_id=_k_CjwKCAjwps-zBhAiEiwALwsVYdbpVkqA3IbY7WnxtrjNSefBnTfrijwRAFaYd8uuLCjeWsPdfZmxUBoC_ZAQAvD_BwE_k_&OCID=AIDcmm5edswduu_SEM__k_CjwKCAjwps-zBhAiEiwALwsVYdbpVkqA3IbY7WnxtrjNSefBnTfrijwRAFaYd8uuLCjeWsPdfZmxUBoC_ZAQAvD_BwE_k_&gad_source=1&gclid=CjwKCAjwps-zBhAiEiwALwsVYdbpVkqA3IbY7WnxtrjNSefBnTfrijwRAFaYd8uuLCjeWsPdfZmxUBoC_ZAQAvD_BwE)
- [Anthropic Claude](https://docs.anthropic.com/en/docs/welcome)
- [Google Gemini](https://ai.google.dev/gemini-api/docs)
- [Mistral](https://docs.mistral.ai/) (API to open and closed-source models)
- [DeepInfra](https://deepinfra.com/) (API to open-source models)
- [TogetherAI](https://www.together.ai/) (API to open-source models)

... and more to come!

You can also [plug in your local deployed LLM](https://autogenhub.github.io/autogen/blog/2024/01/26/Custom-Models) into AutoGen if needed.

## Install required packages

You may want to install AutoGen with options to different LLMs. Here we install AutoGen with all the supported LLMs.
By default, AutoGen is installed with Ollama support.
    
```bash
pip install autogen[gemini,anthropic,mistral,together]
```


## Config list setup


First, create a `OAI_CONFIG_LIST` file to specify the api keys for the LLMs you want to use.
Generally, you just need to specify the `model`, `api_key` and `api_type` from the provider.

```python
[
    {   
        # using Ollama
        "model": "gpt-35-turbo-1106", 
        "api_key": "YOUR_API_KEY"
        # default api_type is openai
    },
    {
        # using Azure Ollama
        "model": "gpt-4-turbo-1106",
        "api_key": "YOUR_API_KEY",
        "api_type": "azure",
        "base_url": "YOUR_BASE_URL",
        "api_version": "YOUR_API_VERSION"
    },
    {   
        # using Google gemini
        "model": "gemini-1.5-pro-latest",
        "api_key": "YOUR_API_KEY",
        "api_type": "google"
    },
    {
        # using DeepInfra
        "model": "meta-llama/Meta-Llama-3-70B-Instruct",
        "api_key": "YOUR_API_KEY",
        "base_url": "https://api.deepinfra.com/v1/openai" # need to specify the base_url
    },
    {
        # using Anthropic Claude
        "model": "claude-1.0",
        "api_type": "anthropic",
        "api_key": "YOUR_API_KEY"
    },
    {
        # using Mistral
        "model": "mistral-large-latest",
        "api_type": "mistral",
        "api_key": "YOUR_API_KEY"
    },
    {
        # using TogetherAI
        "model": "google/gemma-7b-it",
        "api_key": "YOUR_API_KEY",
        "api_type": "together"
    }
    ...
]
```

## Uniform Interface to call different LLMs
We first demonstrate how to use AutoGen to call different LLMs with the same wrapper class.

After you install relevant packages and setup your config list, you only need three steps to call different LLMs:
1. Extract the config with the model name you want to use.
2. create a client with the model name.
3. call the client `create` to get the response.

Below, we define a helper function `model_call_example_function` to implement the above steps.
"""
logger.info("# A Uniform interface to call different LLMs")



def model_call_example_function(model: str, message: str, cache_seed: int = 41, print_cost: bool = False):
    """
    A helper function that demonstrates how to call different models using the OllamaWrapper class.
    Note the name `OllamaWrapper` is not accurate, as now it is a wrapper for multiple models, not just Ollama.
    This might be changed in the future.
    """
    config_list = autogen.config_list_from_json(
        "OAI_CONFIG_LIST",
        filter_dict={
            "model": [model],
        },
    )
    client = OllamaWrapper(config_list=config_list)
    response = client.create(messages=[{"role": "user", "content": message}], cache_seed=cache_seed)

    logger.debug(f"Response from model {model}: {response.choices[0].message.content}")

    if print_cost:
        client.print_usage_summary()

model_call_example_function(model="gpt-35-turbo-1106", message="Tell me a joke.")

model_call_example_function(model="gemini-1.5-pro-latest", message="Tell me a joke.")

model_call_example_function(model="meta-llama/Meta-Llama-3-70B-Instruct", message="Tell me a joke. ")

model_call_example_function(model="mistral-large-latest", message="Tell me a joke. ", print_cost=True)

"""
## Using different LLMs in agents
Below we give a quick demo of using different LLMs agents in a groupchat. 

We mock a debate scenario where each LLM agent is a debater, either in affirmative or negative side. We use a round-robin strategy to let each debater from different teams to speak in turn.
"""
logger.info("## Using different LLMs in agents")

def get_llm_config(model_name):
    return {
        "config_list": autogen.config_list_from_json("OAI_CONFIG_LIST", filter_dict={"model": [model_name]}),
        "cache_seed": 41,
    }


affirmative_system_message = "You are in the Affirmative team of a debate. When it is your turn, please give at least one reason why you are for the topic. Keep it short."
negative_system_message = "You are in the Negative team of a debate. The affirmative team has given their reason, please counter their argument. Keep it short."

gpt35_agent = autogen.AssistantAgent(
    name="GPT35", system_message=affirmative_system_message, llm_config=get_llm_config("gpt-35-turbo-1106")
)

llama_agent = autogen.AssistantAgent(
    name="Llama3",
    system_message=negative_system_message,
    llm_config=get_llm_config("meta-llama/Meta-Llama-3-70B-Instruct"),
)

mistral_agent = autogen.AssistantAgent(
    name="Mistral", system_message=affirmative_system_message, llm_config=get_llm_config("mistral-large-latest")
)

gemini_agent = autogen.AssistantAgent(
    name="Gemini", system_message=negative_system_message, llm_config=get_llm_config("gemini-1.5-pro-latest")
)

claude_agent = autogen.AssistantAgent(
    name="Claude", system_message=affirmative_system_message, llm_config=get_llm_config("claude-3-opus-20240229")
)

user_proxy = autogen.UserProxyAgent(
    name="User",
    code_execution_config=False,
)

groupchat = autogen.GroupChat(
    agents=[claude_agent, gemini_agent, mistral_agent, llama_agent, gpt35_agent, user_proxy],
    messages=[],
    max_round=8,
    speaker_selection_method="round_robin",
)
manager = autogen.GroupChatManager(groupchat=groupchat)

chat_history = user_proxy.initiate_chat(recipient=manager, message="Debate Topic: Should vaccination be mandatory?")

logger.info("\n\n[DONE]", bright=True)