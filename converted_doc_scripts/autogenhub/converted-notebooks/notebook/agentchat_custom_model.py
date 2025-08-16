from autogen import AssistantAgent, UserProxyAgent
from jet.logger import CustomLogger
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from types import SimpleNamespace
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
# Agent Chat with custom model loading

In this notebook, we demonstrate how a custom model can be defined and loaded, and what protocol it needs to comply to.

**NOTE: Depending on what model you use, you may need to play with the default prompts of the Agent's**

## Requirements

````{=mdx}
:::info Requirements
Some extra dependencies are needed for this notebook, which can be installed via pip:

```bash
pip install pyautogen torch transformers sentencepiece
```

For more information, please refer to the [installation guide](/docs/installation/).
:::
````
"""
logger.info("# Agent Chat with custom model loading")




"""
## Create and configure the custom model

A custom model class can be created in many ways, but needs to adhere to the `ModelClient` protocol and response structure which is defined in client.py and shown below.

The response protocol has some minimum requirements, but can be extended to include any additional information that is needed.
Message retrieval therefore can be customized, but needs to return a list of strings or a list of `ModelClientResponseProtocol.Choice.Message` objects.


```python
class ModelClient(Protocol):
    """
logger.info("## Create and configure the custom model")
    A client class must implement the following methods:
    - create must return a response object that implements the ModelClientResponseProtocol
    - cost must return the cost of the response
    - get_usage must return a dict with the following keys:
        - prompt_tokens
        - completion_tokens
        - total_tokens
        - cost
        - model

    This class is used to create a client that can be used by OllamaWrapper.
    The response returned from create must adhere to the ModelClientResponseProtocol but can be extended however needed.
    The message_retrieval method must be implemented to return a list of str or a list of messages from the response.
    """

    RESPONSE_USAGE_KEYS = ["prompt_tokens", "completion_tokens", "total_tokens", "cost", "model"]

    class ModelClientResponseProtocol(Protocol):
        class Choice(Protocol):
            class Message(Protocol):
                content: Optional[str]

            message: Message

        choices: List[Choice]
        model: str

    def create(self, params) -> ModelClientResponseProtocol:
        ...

    def message_retrieval(
        self, response: ModelClientResponseProtocol
    ) -> Union[List[str], List[ModelClient.ModelClientResponseProtocol.Choice.Message]]:
        """
logger.info("RESPONSE_USAGE_KEYS = ["prompt_tokens", "completion_tokens", "total_tokens", "cost", "model"]")
        Retrieve and return a list of strings or a list of Choice.Message from the response.

        NOTE: if a list of Choice.Message is returned, it currently needs to contain the fields of Ollama's ChatCompletion Message object,
        since that is expected for function or tool calling in the rest of the codebase at the moment, unless a custom agent is being used.
        """
        ...

    def cost(self, response: ModelClientResponseProtocol) -> float:
        ...

    @staticmethod
    def get_usage(response: ModelClientResponseProtocol) -> Dict:
        """
logger.info("def cost(self, response: ModelClientResponseProtocol) -> float:")Return usage summary of the response using RESPONSE_USAGE_KEYS."""
        ...
```

## Example of simple custom client

Following the huggingface example for using [Mistral's Open-Orca](https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca)

For the response object, python's `SimpleNamespace` is used to create a simple object that can be used to store the response data, but any object that follows the `ClientResponseProtocol` can be used.
"""
logger.info("## Example of simple custom client")

class CustomModelClient:
    def __init__(self, config, **kwargs):
        logger.debug(f"CustomModelClient config: {config}")
        self.device = config.get("device", "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(config["model"]).to(self.device)
        self.model_name = config["model"]
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"], use_fast=False)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        gen_config_params = config.get("params", {})
        self.max_length = gen_config_params.get("max_length", 256)

        logger.debug(f"Loaded model {config['model']} to {self.device}")

    def create(self, params):
        if params.get("stream", False) and "messages" in params:
            raise NotImplementedError("Local models do not support streaming.")
        else:
            num_of_responses = params.get("n", 1)


            response = SimpleNamespace()

            inputs = self.tokenizer.apply_chat_template(
                params["messages"], return_tensors="pt", add_generation_prompt=True
            ).to(self.device)
            inputs_length = inputs.shape[-1]

            max_length = self.max_length + inputs_length
            generation_config = GenerationConfig(
                max_length=max_length,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            response.choices = []
            response.model = self.model_name

            for _ in range(num_of_responses):
                outputs = self.model.generate(inputs, generation_config=generation_config)
                text = self.tokenizer.decode(outputs[0, inputs_length:])
                choice = SimpleNamespace()
                choice.message = SimpleNamespace()
                choice.message.content = text
                choice.message.function_call = None
                response.choices.append(choice)

            return response

    def message_retrieval(self, response):
        """Retrieve the messages from the response."""
        choices = response.choices
        return [choice.message.content for choice in choices]

    def cost(self, response) -> float:
        """Calculate the cost of the response."""
        response.cost = 0
        return 0

    @staticmethod
    def get_usage(response):
        return {}

"""
## Set your API Endpoint

The [`config_list_from_json`](https://autogenhub.github.io/autogen/docs/reference/oai/openai_utils#config_list_from_json) function loads a list of configurations from an environment variable or a json file.

It first looks for an environment variable of a specified name ("OAI_CONFIG_LIST" in this example), which needs to be a valid json string. If that variable is not found, it looks for a json file with the same name. It filters the configs by models (you can filter by other keys as well).

The json looks like the following:
```json
[
    {
        "model": "gpt-4",
        "api_key": "<your Ollama API key here>"
    },
    {
        "model": "gpt-4",
        "api_key": "<your Azure Ollama API key here>",
        "base_url": "<your Azure Ollama API base here>",
        "api_type": "azure",
        "api_version": "2024-02-01"
    },
    {
        "model": "gpt-4-32k",
        "api_key": "<your Azure Ollama API key here>",
        "base_url": "<your Azure Ollama API base here>",
        "api_type": "azure",
        "api_version": "2024-02-01"
    }
]
```

You can set the value of config_list in any way you prefer. Please refer to this [notebook](https://github.com/microsoft/autogen/blob/main/notebook/oai_openai_utils.ipynb) for full code examples of the different methods.

## Set the config for the custom model

You can add any paramteres that are needed for the custom model loading in the same configuration list.

It is important to add the `model_client_cls` field and set it to a string that corresponds to the class name: `"CustomModelClient"`.

```json
{
    "model": "Open-Orca/Mistral-7B-OpenOrca",
    "model_client_cls": "CustomModelClient",
    "device": "cuda",
    "n": 1,
    "params": {
        "max_length": 1000,
    }
},
```
"""
logger.info("## Set your API Endpoint")

config_list_custom = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={"model_client_cls": ["CustomModelClient"]},
)

"""
## Construct Agents

Consturct a simple conversation between a User proxy and an Assistent agent
"""
logger.info("## Construct Agents")

assistant = AssistantAgent("assistant", llm_config={"config_list": config_list_custom})
user_proxy = UserProxyAgent(
    "user_proxy",
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    },
)

"""
## Register the custom client class to the assistant agent
"""
logger.info("## Register the custom client class to the assistant agent")

assistant.register_model_client(model_client_cls=CustomModelClient)

user_proxy.initiate_chat(assistant, message="Write python code to print Hello World!")

"""
## Register a custom client class with a pre-loaded model

If you want to have more control over when the model gets loaded, you can load the model yourself and pass it as an argument to the CustomClient during registration
"""
logger.info("## Register a custom client class with a pre-loaded model")

class CustomModelClientWithArguments(CustomModelClient):
    def __init__(self, config, loaded_model, tokenizer, **kwargs):
        logger.debug(f"CustomModelClientWithArguments config: {config}")

        self.model_name = config["model"]
        self.model = loaded_model
        self.tokenizer = tokenizer

        self.device = config.get("device", "cpu")

        gen_config_params = config.get("params", {})
        self.max_length = gen_config_params.get("max_length", 256)
        logger.debug(f"Loaded model {config['model']} to {self.device}")

config = config_list_custom[0]
device = config.get("device", "cpu")
loaded_model = AutoModelForCausalLM.from_pretrained(config["model"]).to(device)
tokenizer = AutoTokenizer.from_pretrained(config["model"], use_fast=False)
tokenizer.pad_token_id = tokenizer.eos_token_id

"""
## Add the config of the new custom model

```json
{
    "model": "Open-Orca/Mistral-7B-OpenOrca",
    "model_client_cls": "CustomModelClientWithArguments",
    "device": "cuda",
    "n": 1,
    "params": {
        "max_length": 1000,
    }
},
```
"""
logger.info("## Add the config of the new custom model")

config_list_custom = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={"model_client_cls": ["CustomModelClientWithArguments"]},
)

assistant = AssistantAgent("assistant", llm_config={"config_list": config_list_custom})

assistant.register_model_client(
    model_client_cls=CustomModelClientWithArguments,
    loaded_model=loaded_model,
    tokenizer=tokenizer,
)

user_proxy.initiate_chat(assistant, message="Write python code to print Hello World!")

logger.info("\n\n[DONE]", bright=True)