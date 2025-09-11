from jet.logger import logger
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.pydantic_v1 import BaseTool
from pydantic import BaseModel
from pydantic import Field, field_validator # pydantic v2
from pydantic.v1 import validator, Field # if pydantic 2 is installed
from typing import Optional as Optional
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
# LangChain v0.3

*Last updated: 09.16.24*

## What's changed

* All packages have been upgraded from Pydantic 1 to Pydantic 2 internally. Use of Pydantic 2 in user code is fully supported with all packages without the need for bridges like `langchain_core.pydantic_v1` or `pydantic.v1`.
* Pydantic 1 will no longer be supported as it reached its end-of-life in June 2024.
* Python 3.8 will no longer be supported as its end-of-life is October 2024.

**These are the only breaking changes.**

## What's new

The following features have been added during the development of 0.2.x:

- Moved more integrations from `langchain-community` to their own `langchain-x` packages. This is a non-breaking change, as the legacy implementations are left in `langchain-community` and marked as deprecated. This allows us to better manage the dependencies of, test, and version these integrations. You can see all the latest integration packages in the [API reference](https://python.langchain.com/v0.2/api_reference/reference.html#integrations).
- Simplified tool definition and usage. Read more [here](https://blog.langchain.dev/improving-core-tool-interfaces-and-docs-in-langchain/).
- Added utilities for interacting with chat models: [universal model constructor](https://python.langchain.com/v0.2/docs/how_to/chat_models_universal_init/), [rate limiter](https://python.langchain.com/v0.2/docs/how_to/chat_model_rate_limiting/), [message utilities](https://python.langchain.com/v0.2/docs/how_to/#messages),
- Added the ability to [dispatch custom events](https://python.langchain.com/v0.2/docs/how_to/callbacks_custom_events/).
- Revamped integration docs and API reference. Read more [here](https://blog.langchain.dev/langchain-integration-docs-revamped/).
- Marked as deprecated a number of legacy chains and added migration guides for all of them. These are slated for removal in `langchain` 1.0.0. See the deprecated chains and associated [migration guides here](https://python.langchain.com/v0.2/docs/versions/migrating_chains/).

## How to update your code

If you're using `langchain` / `langchain-community` / `langchain-core` 0.0 or 0.1, we recommend that you first [upgrade to 0.2](https://python.langchain.com/v0.2/docs/versions/v0_2/).

If you're using `langgraph`, upgrade to `langgraph>=0.2.20,<0.3`. This will work with either 0.2 or 0.3 versions of all the base packages.

Here is a complete list of all packages that have been released and what we recommend upgrading your version constraints to.
Any package that now requires `langchain-core` 0.3 had a minor version bump.
Any package that is now compatible with both `langchain-core` 0.2 and 0.3 had a patch version bump.

You can use the `langchain-cli` to update deprecated imports automatically.
The CLI will handle updating deprecated imports that were introduced in LangChain 0.0.x and LangChain 0.1, as
well as updating the `langchain_core.pydantic_v1` and `langchain.pydantic_v1` imports.


### Base packages

| Package                  | Latest | Recommended constraint |
|--------------------------|--------|------------------------|
| langchain                | 0.3.0  | >=0.3,&lt;0.4             |
| langchain-community      | 0.3.0  | >=0.3,&lt;0.4             |
| langchain-text-splitters | 0.3.0  | >=0.3,&lt;0.4             |
| langchain-core           | 0.3.0  | >=0.3,&lt;0.4             |
| langchain-experimental   | 0.3.0  | >=0.3,&lt;0.4             |

### Downstream packages

| Package   | Latest | Recommended constraint |
|-----------|--------|------------------------|
| langgraph | 0.2.20 | >=0.2.20,&lt;0.3          |
| langserve | 0.3.0  | >=0.3,&lt;0.4             |

### Integration packages

| Package                                | Latest  | Recommended constraint     |
| -------------------------------------- | ------- | -------------------------- |
| langchain-ai21                         | 0.2.0   | >=0.2,&lt;0.3                 |
| langchain-aws                          | 0.2.0   | >=0.2,&lt;0.3                 |
| langchain-anthropic                    | 0.2.0   | >=0.2,&lt;0.3                 |
| langchain-astradb                      | 0.4.1   | >=0.4.1,&lt;0.5               |
| langchain-azure-dynamic-sessions       | 0.2.0   | >=0.2,&lt;0.3                 |
| langchain-box                          | 0.2.0   | >=0.2,&lt;0.3                 |
| langchain-chroma                       | 0.1.4   | >=0.1.4,&lt;0.2               |
| langchain-cohere                       | 0.3.0   | >=0.3,&lt;0.4                 |
| langchain-elasticsearch                | 0.3.0   | >=0.3,&lt;0.4                 |
| langchain-exa                          | 0.2.0   | >=0.2,&lt;0.3                 |
| langchain-fireworks                    | 0.2.0   | >=0.2,&lt;0.3                 |
| langchain-groq                         | 0.2.0   | >=0.2,&lt;0.3                 |
| langchain-google-community             | 2.0.0   | >=2,&lt;3                     |
| langchain-google-genai                 | 2.0.0   | >=2,&lt;3                     |
| langchain-google-vertexai              | 2.0.0   | >=2,&lt;3                     |
| langchain-huggingface                  | 0.1.0   | >=0.1,&lt;0.2                 |
| langchain-ibm                          | 0.3.0   | >=0.3,&lt;0.4                 |
| langchain-milvus                       | 0.1.6   | >=0.1.6,&lt;0.2               |
| langchain-mistralai                    | 0.2.0   | >=0.2,&lt;0.3                 |
| langchain-mongodb                      | 0.2.0   | >=0.2,&lt;0.3                 |
| langchain-nomic                        | 0.1.3   | >=0.1.3,&lt;0.2               |
| langchain-nvidia                       | 0.3.0   | >=0.3,&lt;0.4                 |
| langchain-ollama                       | 0.2.0   | >=0.2,&lt;0.3                 |
| langchain-ollama                       | 0.2.0   | >=0.2,&lt;0.3                 |
| langchain-pinecone                     | 0.2.0   | >=0.2,&lt;0.3                 |
| langchain-postgres                     | 0.0.13  | >=0.0.13,&lt;0.1              |
| langchain-prompty                      | 0.1.0   | >=0.1,&lt;0.2                 |
| langchain-qdrant                       | 0.1.4   | >=0.1.4,&lt;0.2               |
| langchain-redis                        | 0.1.0   | >=0.1,&lt;0.2                 |
| langchain-sema4                        | 0.2.0   | >=0.2,&lt;0.3                 |
| langchain-together                     | 0.2.0   | >=0.2,&lt;0.3                 |
| langchain-unstructured                 | 0.1.4   | >=0.1.4,&lt;0.2               |
| langchain-upstage                      | 0.3.0   | >=0.3,&lt;0.4                 |
| langchain-voyageai                     | 0.2.0   | >=0.2,&lt;0.3                 |
| langchain-weaviate                     | 0.0.3   | >=0.0.3,&lt;0.1               |

Once you've updated to recent versions of the packages, you may need to address the following issues stemming from the internal switch from Pydantic v1 to Pydantic v2:

- If your code depends on Pydantic aside from LangChain, you will need to upgrade your pydantic version constraints to be `pydantic>=2,<3`.  See [Pydantic's migration guide](https://docs.pydantic.dev/latest/migration/) for help migrating your non-LangChain code to Pydantic v2 if you use pydantic v1.
- There are a number of side effects to LangChain components caused by the internal switch from Pydantic v1 to v2. We have listed some of the common cases below together with the recommended solutions.

## Common issues when transitioning to Pydantic 2

### 1. Do not use the `langchain_core.pydantic_v1` namespace

Replace any usage of `langchain_core.pydantic_v1` or `langchain.pydantic_v1` with
direct imports from `pydantic`.

For example,
"""
logger.info("# LangChain v0.3")


"""
to:
"""
logger.info("to:")


"""
This may require you to make additional updates to your Pydantic code given that there are a number of breaking changes in Pydantic 2. See the [Pydantic Migration](https://docs.pydantic.dev/latest/migration/) for how to upgrade your code from Pydantic 1 to 2.

### 2. Passing Pydantic objects to LangChain APIs

Users using the following APIs:

* `BaseChatModel.bind_tools`
* `BaseChatModel.with_structured_output`
* `Tool.from_function`
* `StructuredTool.from_function`

should ensure that they are passing Pydantic 2 objects to these APIs rather than
Pydantic 1 objects (created via the `pydantic.v1` namespace of pydantic 2).

:::caution
While `v1` objects may be accepted by some of these APIs, users are advised to
use Pydantic 2 objects to avoid future issues.
:::

### 3. Sub-classing LangChain models

Any sub-classing from existing LangChain models (e.g., `BaseTool`, `BaseChatModel`, `LLM`)
should upgrade to use Pydantic 2 features.

For example, any user code that's relying on Pydantic 1 features (e.g., `validator`) should
be updated to the Pydantic 2 equivalent (e.g., `field_validator`), and any references to
`pydantic.v1`, `langchain_core.pydantic_v1`, `langchain.pydantic_v1` should be replaced
with imports from `pydantic`.
"""
logger.info("### 2. Passing Pydantic objects to LangChain APIs")


class CustomTool(BaseTool): # BaseTool is v1 code
    x: int = Field(default=1)

    def _run(*args, **kwargs):
        return "hello"

    @validator('x') # v1 code
    @classmethod
    def validate_x(cls, x: int) -> int:
        return 1

"""
Should change to:
"""
logger.info("Should change to:")


class CustomTool(BaseTool): # BaseTool is v1 code
    x: int = Field(default=1)

    def _run(*args, **kwargs):
        return "hello"

    @field_validator('x') # v2 code
    @classmethod
    def validate_x(cls, x: int) -> int:
        return 1


CustomTool(
    name='custom_tool',
    description="hello",
    x=1,
)

"""
### 4. model_rebuild()

When sub-classing from LangChain models, users may need to add relevant imports
to the file and rebuild the model.

You can read more about `model_rebuild` [here](https://docs.pydantic.dev/latest/concepts/models/#rebuilding-model-schema).
"""
logger.info("### 4. model_rebuild()")



class FooParser(BaseOutputParser):
    ...

"""
New code:
"""
logger.info("New code:")



class FooParser(BaseOutputParser):
    ...

FooParser.model_rebuild()

"""
## Migrate using langchain-cli

The `langchain-cli` can help update deprecated LangChain imports in your code automatically.

Please note that the `langchain-cli` only handles deprecated LangChain imports and cannot
help to upgrade your code from pydantic 1 to pydantic 2.

For help with the Pydantic 1 to 2 migration itself please refer to the [Pydantic Migration Guidelines](https://docs.pydantic.dev/latest/migration/).

As of 0.0.31, the `langchain-cli` relies on [gritql](https://about.grit.io/) for applying code mods.

### Installation
"""
logger.info("## Migrate using langchain-cli")

pip install -U langchain-cli
langchain-cli --version # <-- Make sure the version is at least 0.0.31

"""
### Usage

Given that the migration script is not perfect, you should make sure you have a backup of your code first (e.g., using version control like `git`).

The `langchain-cli` will handle the `langchain_core.pydantic_v1` deprecation introduced in LangChain 0.3 as well
as older deprecations (e.g.,`from langchain.chat_models import ChatOllama` which should be `from jet.adapters.langchain.chat_ollama import ChatOllama`),

You will need to run the migration script **twice** as it only applies one import replacement per run.

For example, say that your code is still using the old import `from langchain.chat_models import ChatOllama`:

After the first run, you'll get: `from langchain_community.chat_models import ChatOllama`
After the second run, you'll get: `from jet.adapters.langchain.chat_ollama import ChatOllama`
"""
logger.info("### Usage")

langchain-cli migrate --help [path to code] # Help
langchain-cli migrate [path to code] # Apply

langchain-cli migrate --diff [path to code] # Preview
langchain-cli migrate [path to code] # Apply

"""
### Other options
"""
logger.info("### Other options")

langchain-cli migrate --help
langchain-cli migrate --diff [path to code]
langchain-cli migrate --interactive [path to code]

logger.info("\n\n[DONE]", bright=True)