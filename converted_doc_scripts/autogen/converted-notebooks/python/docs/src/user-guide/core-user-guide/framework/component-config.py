from autogen_core import Component, ComponentBase
from autogen_core.models import ChatCompletionClient
from jet.logger import CustomLogger
from pydantic import BaseModel
from pydantic import BaseModel, SecretStr
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")


"""
# Component config

AutoGen components are able to be declaratively configured in a generic fashion. This is to support configuration based experiences, such as AutoGen studio, but it is also useful for many other scenarios.

The system that provides this is called "component configuration". In AutoGen, a component is simply something that can be created from a config object and itself can be dumped to a config object. In this way, you can define a component in code and then get the config object from it.

This system is generic and allows for components defined outside of AutoGen itself (such as extensions) to be configured in the same way.

## How does this differ from state?

This is a very important point to clarify. When we talk about serializing an object, we must include *all* data that makes that object itself. Including things like message history etc. When deserializing from serialized state, you must get back the *exact* same object. This is not the case with component configuration.

Component configuration should be thought of as the blueprint for an object, and can be stamped out many times to create many instances of the same configured object.

## Usage

If you have a component in Python and want to get the config for it, simply call {py:meth}`~autogen_core.ComponentToConfig.dump_component` on it. The resulting object can be passed back into {py:meth}`~autogen_core.ComponentLoader.load_component` to get the component back.

### Loading a component from a config

To load a component from a config object, you can use the {py:meth}`~autogen_core.ComponentLoader.load_component` method. This method will take a config object and return a component object. It is best to call this method on the interface you want. For example to load a model client:
"""
logger.info("# Component config")


config = {
    "provider": "openai_chat_completion_client",
    "config": {"model": "qwen3-1.7b-4bit"},
}

client = ChatCompletionClient.load_component(config)

"""
## Creating a component class

To add component functionality to a given class:

1. Add a call to {py:meth}`~autogen_core.Component` in the class inheritance list.
2. Implment the {py:meth}`~autogen_core.ComponentToConfig._to_config` and {py:meth}`~autogen_core.ComponentFromConfig._from_config` methods

For example:
"""
logger.info("## Creating a component class")



class Config(BaseModel):
    value: str


class MyComponent(ComponentBase[Config], Component[Config]):
    component_type = "custom"
    component_config_schema = Config

    def __init__(self, value: str):
        self.value = value

    def _to_config(self) -> Config:
        return Config(value=self.value)

    @classmethod
    def _from_config(cls, config: Config) -> "MyComponent":
        return cls(value=config.value)

"""
## Secrets

If a field of a config object is a secret value, it should be marked using [`SecretStr`](https://docs.pydantic.dev/latest/api/types/#pydantic.types.SecretStr), this will ensure that the value will not be dumped to the config object.

For example:
"""
logger.info("## Secrets")



class ClientConfig(BaseModel):
    endpoint: str
    api_key: SecretStr

logger.info("\n\n[DONE]", bright=True)