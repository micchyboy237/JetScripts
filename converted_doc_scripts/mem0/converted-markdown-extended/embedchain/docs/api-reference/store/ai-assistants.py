from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: 'AI Assistant'
---

The `AIAssistant` class, an alternative to the MLX Assistant API, is designed for those who prefer using large language models (LLMs) other than those provided by MLX. It facilitates the creation of AI Assistants with several key benefits:

- **Visibility into Citations**: It offers transparent access to the sources and citations used by the AI, enhancing the understanding and trustworthiness of its responses.

- **Debugging Capabilities**: Users have the ability to delve into and debug the AI's processes, allowing for a deeper understanding and fine-tuning of its performance.

- **Customizable Prompts**: The class provides the flexibility to modify and tailor prompts according to specific needs, enabling more precise and relevant interactions.

- **Chain of Thought Integration**: It supports the incorporation of a 'chain of thought' approach, which helps in breaking down complex queries into simpler, sequential steps, thereby improving the clarity and accuracy of responses.

It is ideal for those who value customization, transparency, and detailed control over their AI Assistant's functionalities.

### Arguments

<ParamField path="name" type="string" optional>
  Name for your AI assistant
</ParamField>

<ParamField path="instructions" type="string" optional>
  How the Assistant and model should behave or respond
</ParamField>

<ParamField path="assistant_id" type="string" optional>
  Load existing AI Assistant. If you pass this, you don't have to pass other arguments.
</ParamField>

<ParamField path="thread_id" type="string" optional>
  Existing thread id if exists
</ParamField>

<ParamField path="yaml_path" type="str" Optional>
    Embedchain pipeline config yaml path to use. This will define the configuration of the AI Assistant (such as configuring the LLM, vector database, and embedding model)
</ParamField>

<ParamField path="data_sources" type="list" default="[]">
  Add data sources to your assistant. You can add in the following format: `[{"source": "https://example.com", "data_type": "web_page"}]`
</ParamField>

<ParamField path="collect_metrics" type="boolean" default="True">
  Anonymous telemetry (doesn't collect any user information or user's files). Used to improve the Embedchain package utilization. Default is `True`.
</ParamField>


## Usage

For detailed guidance on creating your own AI Assistant, click the link below. It provides step-by-step instructions to help you through the process:

<Card title="Guide to Creating Your AI Assistant" icon="link" href="/examples/opensource-assistant">
  Learn how to build a customized AI Assistant using the `AIAssistant` class.
</Card>
"""
logger.info("### Arguments")

logger.info("\n\n[DONE]", bright=True)