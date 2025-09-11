from jet.logger import logger
from langchain_aws import BedrockLLM
from langchain_core.callbacks import AsyncCallbackHandler
from typing import Any
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
# Bedrock

:::caution
You are currently on a page documenting the use of Amazon Bedrock models as [text completion models](/docs/concepts/text_llms). Many popular models available on Bedrock are [chat completion models](/docs/concepts/chat_models).

You may be looking for [this page instead](/docs/integrations/chat/bedrock/).
:::

>[Amazon Bedrock](https://aws.amazon.com/bedrock/) is a fully managed service that offers a choice of 
> high-performing foundation models (FMs) from leading AI companies like `AI21 Labs`, `Ollama`, `Cohere`, 
> `Meta`, `Stability AI`, and `Amazon` via a single API, along with a broad set of capabilities you need to 
> build generative AI applications with security, privacy, and responsible AI. Using `Amazon Bedrock`, 
> you can easily experiment with and evaluate top FMs for your use case, privately customize them with 
> your data using techniques such as fine-tuning and `Retrieval Augmented Generation` (`RAG`), and build 
> agents that execute tasks using your enterprise systems and data sources. Since `Amazon Bedrock` is 
> serverless, you don't have to manage any infrastructure, and you can securely integrate and deploy 
> generative AI capabilities into your applications using the AWS services you are already familiar with.
"""
logger.info("# Bedrock")

# %pip install --upgrade --quiet langchain-aws


llm = BedrockLLM(
    credentials_profile_name="bedrock-admin", model_id="amazon.titan-text-express-v1"
)

"""
### Custom models
"""
logger.info("### Custom models")

custom_llm = BedrockLLM(
    credentials_profile_name="bedrock-admin",
    provider="cohere",
    model_id="<Custom model ARN>",  # ARN like 'arn:aws:bedrock:...' obtained via provisioning the custom model
    model_kwargs={"temperature": 1},
    streaming=True,
)

custom_llm.invoke(input="What is the recipe of mayonnaise?")

"""
## Guardrails for Amazon Bedrock

[Guardrails for Amazon Bedrock](https://aws.amazon.com/bedrock/guardrails/) evaluates user inputs and model responses based on use case specific policies, and provides an additional layer of safeguards regardless of the underlying model. Guardrails can be applied across models, including Ollama Claude, Meta Llama 2, Cohere Command, AI21 Labs Jurassic, and Amazon Titan Text, as well as fine-tuned models.
**Note**: Guardrails for Amazon Bedrock is currently in preview and not generally available. Reach out through your usual AWS Support contacts if you’d like access to this feature.
In this section, we are going to set up a Bedrock language model with specific guardrails that include tracing capabilities.
"""
logger.info("## Guardrails for Amazon Bedrock")




class BedrockAsyncCallbackHandler(AsyncCallbackHandler):

    async def on_llm_error(self, error: BaseException, **kwargs: Any) -> Any:
        reason = kwargs.get("reason")
        if reason == "GUARDRAIL_INTERVENED":
            logger.debug(f"Guardrails: {kwargs}")


llm = BedrockLLM(
    credentials_profile_name="bedrock-admin",
    model_id="<Model_ID>",
    model_kwargs={},
    guardrails={"id": "<Guardrail_ID>", "version": "<Version>", "trace": True},
    callbacks=[BedrockAsyncCallbackHandler()],
)

logger.info("\n\n[DONE]", bright=True)