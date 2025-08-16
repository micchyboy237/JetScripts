from autogen import ConversableAgent, UserProxyAgent
from jet.logger import CustomLogger
from typing import Literal
from typing_extensions import Annotated
import autogen
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# LiteLLM with Ollama
[LiteLLM](https://litellm.ai/) is an open-source locally run proxy server that provides an
MLX-compatible API. It interfaces with a large number of providers that do the inference.
To handle the inference, a popular open-source inference engine is [Ollama](https://ollama.com/).

As not all proxy servers support MLX's [Function Calling](https://platform.openai.com/docs/guides/function-calling) (usable with AutoGen),
LiteLLM together with Ollama enable this useful feature.

Running this stack requires the installation of:

1. AutoGen ([installation instructions](/docs/installation))
2. LiteLLM
3. Ollama

Note: We recommend using a virtual environment for your stack, see [this article](https://autogenhub.github.io/autogen/docs/installation/#create-a-virtual-environment-optional) for guidance.

## Installing LiteLLM

Install LiteLLM with the proxy server functionality:

```bash
pip install 'litellm[proxy]'
```

Note: If using Windows, run LiteLLM and Ollama within a [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install).

````mdx-code-block
:::tip
For custom LiteLLM installation instructions, see their [GitHub repository](https://github.com/BerriAI/litellm).
:::
````

## Installing Ollama

For Mac and Windows, [download Ollama](https://ollama.com/download).

For Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## Downloading models

Ollama has a library of models to choose from, see them [here](https://ollama.com/library).

Before you can use a model, you need to download it (using the name of the model from the library):

```bash
ollama pull llama3:instruct
```

To view the models you have downloaded and can use:

```bash
ollama list
```

````mdx-code-block
:::tip
Ollama enables the use of GGUF model files, available readily on Hugging Face. See Ollama`s [GitHub repository](https://github.com/ollama/ollama)
for examples.
:::
````

## Running LiteLLM proxy server

To run LiteLLM with the model you have downloaded, in your terminal:

```bash
litellm --model ollama/llama3:instruct
```

```` text
INFO:     Started server process [19040]
INFO:     Waiting for application startup.

#------------------------------------------------------------#
#                                                            #
#       'This feature doesn't meet my needs because...'       #
#        https://github.com/BerriAI/litellm/issues/new        #
#                                                            #
#------------------------------------------------------------#

 Thank you for using LiteLLM! - Krrish & Ishaan



Give Feedback / Get Help: https://github.com/BerriAI/litellm/issues/new


INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:4000 (Press CTRL+C to quit)
````

This will run the proxy server and it will be available at 'http://0.0.0.0:4000/'.

## Using LiteLLM+Ollama with AutoGen

Now that we have the URL for the LiteLLM proxy server, you can use it within AutoGen
in the same way as MLX or cloud-based proxy servers.

As you are running this proxy server locally, no API key is required. Additionally, as
the model is being set when running the
LiteLLM command, no model name needs to be configured in AutoGen. However, ```model```
and ```api_key``` are mandatory fields for configurations within AutoGen so we put dummy
values in them, as per the example below.

An additional setting for the configuration is `price`, which can be used to set the pricing of tokens. As we're running it locally, we'll put our costs as zero. Using this setting will also avoid a prompt being shown when price can't be determined.
"""
logger.info("# LiteLLM with Ollama")


local_llm_config = {
    "config_list": [
        {
            "model": "NotRequired",  # Loaded with LiteLLM command
            "api_key": "NotRequired",  # Not needed
            "base_url": "http://0.0.0.0:4000",  # Your LiteLLM URL
            "price": [0, 0],  # Put in price per 1K tokens [prompt, response] as free!
        }
    ],
    "cache_seed": None,  # Turns off caching, useful for testing different models
}

assistant = ConversableAgent("agent", llm_config=local_llm_config)

user_proxy = UserProxyAgent("user", code_execution_config=False)

res = assistant.initiate_chat(user_proxy, message="How can I help you today?")

logger.debug(assistant)

"""
## Example with Function Calling
Function calling (aka Tool calling) is a feature of MLX's API that AutoGen, LiteLLM, and Ollama support.

Below is an example of using function calling with LiteLLM and Ollama. Based on this [currency conversion](https://github.com/microsoft/autogen/blob/501f8d22726e687c55052682c20c97ce62f018ac/notebook/agentchat_function_call_currency_calculator.ipynb) notebook.

LiteLLM is loaded in the same way as the previous example and we'll continue to use Meta's Llama3 model as it is good at constructing the
function calling message required.

**Note:** LiteLLM version 1.41.27, or later, is required (to support function calling natively using Ollama).

In your terminal:

```bash
litellm --model ollama/llama3
```

Then we run our program with function calling.
"""
logger.info("## Example with Function Calling")




local_llm_config = {
    "config_list": [
        {
            "model": "NotRequired",  # Loaded with LiteLLM command
            "api_key": "NotRequired",  # Not needed
            "base_url": "http://0.0.0.0:4000",  # Your LiteLLM URL
            "price": [0, 0],  # Put in price per 1K tokens [prompt, response] as free!
        }
    ],
    "cache_seed": None,  # Turns off caching, useful for testing different models
}

chatbot = autogen.AssistantAgent(
    name="chatbot",
    system_message="""For currency exchange tasks,
        only use the functions you have been provided with.
        If the function has been called previously,
        return only the word 'TERMINATE'.""",
    llm_config=local_llm_config,
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and "TERMINATE" in x.get("content", ""),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
    code_execution_config={"work_dir": "code", "use_docker": False},
)

CurrencySymbol = Literal["USD", "EUR"]



def exchange_rate(base_currency: CurrencySymbol, quote_currency: CurrencySymbol) -> float:
    if base_currency == quote_currency:
        return 1.0
    elif base_currency == "USD" and quote_currency == "EUR":
        return 1 / 1.1
    elif base_currency == "EUR" and quote_currency == "USD":
        return 1.1
    else:
        raise ValueError(f"Unknown currencies {base_currency}, {quote_currency}")


@user_proxy.register_for_execution()
@chatbot.register_for_llm(description="Currency exchange calculator.")
def currency_calculator(
    base_amount: Annotated[float, "Amount of currency in base_currency"],
    base_currency: Annotated[CurrencySymbol, "Base currency"] = "USD",
    quote_currency: Annotated[CurrencySymbol, "Quote currency"] = "EUR",
) -> str:
    quote_amount = exchange_rate(base_currency, quote_currency) * base_amount
    return f"{format(quote_amount, '.2f')} {quote_currency}"

res = user_proxy.initiate_chat(
    chatbot,
    message="How much is 123.45 EUR in USD?",
    summary_method="reflection_with_llm",
)

"""
We can see that the currency conversion function was called with the correct values and a result was generated.

````mdx-code-block
:::tip
Once functions are included in the conversation it is possible, using LiteLLM and Ollama, that the model may continue to recommend tool calls (as shown above). This is an area of active development and a native Ollama client for AutoGen is planned for a future release.
:::
````
"""
logger.info("We can see that the currency conversion function was called with the correct values and a result was generated.")

logger.info("\n\n[DONE]", bright=True)