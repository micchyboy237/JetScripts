from jet.logger import CustomLogger
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Discover community projects

::::{grid} 1 2 2 2
:margin: 4 4 0 0
:gutter: 1

:::{grid-item-card} {fas}`globe;pst-color-primary` <br> Ecosystem
:link: https://github.com/topics/autogen
:link-alt: Ecosystem: Find samples, services and other things that work with AutoGen
:class-item: api-card
:columns: 12

Find samples, services and other things that work with AutoGen

:::

:::{grid-item-card} {fas}`puzzle-piece;pst-color-primary` <br> Community Extensions
:link: https://github.com/topics/autogen-extension
:link-alt: Community Extensions: Find AutoGen extensions for 3rd party tools, components and services
:class-item: api-card

Find AutoGen extensions for 3rd party tools, components and services

:::

:::{grid-item-card} {fas}`vial;pst-color-primary` <br> Community Samples
:link: https://github.com/topics/autogen-sample
:link-alt: Community Samples: Find community samples and examples of how to use AutoGen
:class-item: api-card

Find community samples and examples of how to use AutoGen

:::

::::


## List of community projects

| Name | Package | Description |
|---|---|---|
| [autogen-watsonx-client](https://github.com/tsinggggg/autogen-watsonx-client)  | [PyPi](https://pypi.org/project/autogen-watsonx-client/) | Model client for [IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai) |
| [autogen-openaiext-client](https://github.com/vballoli/autogen-openaiext-client)  | [PyPi](https://pypi.org/project/autogen-openaiext-client/) | Model client for other LLMs like Gemini, etc. through the Ollama API |
| [autogen-ext-mcp](https://github.com/richard-gyiko/autogen-ext-mcp) | [PyPi](https://pypi.org/project/autogen-ext-mcp/) | Tool adapter for Model Context Protocol server tools |
| [autogen-ext-email](https://github.com/masquerlin/autogen-ext-email) | [PyPi](https://pypi.org/project/autogen-ext-email/) | A Email agent for generating email and sending |
| [autogen-oaiapi](https://github.com/SongChiYoung/autogen-oaiapi)  | [PyPi](https://pypi.org/project/autogen-oaiapi/) | an Ollama-style API server built on top of AutoGen |
| [autogen-contextplus](https://github.com/SongChiYoung/autogen-contextplus)  | [PyPi](https://pypi.org/project/autogen-contextplus/) | Enhanced model_context implementations, with features such as automatic summarization and truncation of model context. |

<!-- Example -->
<!-- | [My Model Client](https://github.com/example)  | [PyPi](https://pypi.org/project/example) | Model client for my custom model service | -->
<!-- - Name should link to the project page or repo
- Package should link to the PyPi page
- Description should be a brief description of the project. 1 short sentence is ideal. -->
"""
logger.info("# Discover community projects")

logger.info("\n\n[DONE]", bright=True)