from jet.logger import logger
from langchain_community.chat_models import ChatOCIModelDeployment
from langchain_community.chat_models import ChatOCIModelDeploymentVLLM
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
import ads
import os
import shutil
import sys

async def main():
    
    
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
    ---
    sidebar_label: ChatOCIModelDeployment
    ---
    
    # ChatOCIModelDeployment
    
    This will help you get started with OCIModelDeployment [chat models](/docs/concepts/chat_models). For detailed documentation of all ChatOCIModelDeployment features and configurations head to the [API reference](https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.oci_data_science.ChatOCIModelDeployment.html).
    
    [OCI Data Science](https://docs.oracle.com/en-us/iaas/data-science/using/home.htm) is a fully managed and serverless platform for data science teams to build, train, and manage machine learning models in the Oracle Cloud Infrastructure. You can use [AI Quick Actions](https://blogs.oracle.com/ai-and-datascience/post/ai-quick-actions-in-oci-data-science) to easily deploy LLMs on [OCI Data Science Model Deployment Service](https://docs.oracle.com/en-us/iaas/data-science/using/model-dep-about.htm). You may choose to deploy the model with popular inference frameworks such as vLLM or TGI. By default, the model deployment endpoint mimics the Ollama API protocol.
    
    > For the latest updates, examples and experimental features, please see [ADS LangChain Integration](https://accelerated-data-science.readthedocs.io/en/latest/user_guide/large_language_model/langchain_models.html).
    
    ## Overview
    ### Integration details
    
    | Class | Package | Local | Serializable | JS support | Package downloads | Package latest |
    | :--- | :--- | :---: | :---: |  :---: | :---: | :---: |
    | [ChatOCIModelDeployment](https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.oci_data_science.ChatOCIModelDeployment.html) | [langchain-community](https://python.langchain.com/api_reference/community/index.html) | ❌ | beta | ❌ | ![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-community?style=flat-square&label=%20) | ![PyPI - Version](https://img.shields.io/pypi/v/langchain-community?style=flat-square&label=%20) |
    
    ### Model features
    
    | [Tool calling](/docs/how_to/tool_calling) | [Structured output](/docs/how_to/structured_output/) | JSON mode | [Image input](/docs/how_to/multimodal_inputs/) | Audio input | Video input | [Token-level streaming](/docs/how_to/chat_streaming/) | Native async | [Token usage](/docs/how_to/chat_token_usage_tracking/) | [Logprobs](/docs/how_to/logprobs/) |
    | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
    | depends | depends | depends | depends | depends | depends | ✅ | ✅ | ✅ | ✅ | 
    
    Some model features, including tool calling, structured output, JSON mode and multi-modal inputs, are depending on deployed model.
    
    
    ## Setup
    
    To use ChatOCIModelDeployment you'll need to deploy a chat model with chat completion endpoint and install the `langchain-community`, `langchain-ollama` and `oracle-ads` integration packages.
    
    You can easily deploy foundation models using the [AI Quick Actions](https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/ai-quick-actions/model-deployment-tips.md) on OCI Data Science Model deployment. For additional deployment examples, please visit the [Oracle GitHub samples repository](https://github.com/oracle-samples/oci-data-science-ai-samples/tree/main/ai-quick-actions).
    
    ### Policies
    Make sure to have the required [policies](https://docs.oracle.com/en-us/iaas/data-science/using/model-dep-policies-auth.htm#model_dep_policies_auth__predict-endpoint) to access the OCI Data Science Model Deployment endpoint.
    
    ### Credentials
    
    You can set authentication through Oracle ADS. When you are working in OCI Data Science Notebook Session, you can leverage resource principal to access other OCI resources.
    """
    logger.info("# ChatOCIModelDeployment")
    
    
    ads.set_auth("resource_principal")
    
    """
    Alternatively, you can configure the credentials using the following environment variables. For example, to use API key with specific profile:
    """
    logger.info("Alternatively, you can configure the credentials using the following environment variables. For example, to use API key with specific profile:")
    
    
    os.environ["OCI_IAM_TYPE"] = "api_key"
    os.environ["OCI_CONFIG_PROFILE"] = "default"
    os.environ["OCI_CONFIG_LOCATION"] = "~/.oci"
    
    """
    Check out [Oracle ADS docs](https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/authentication.html) to see more options.
    
    ### Installation
    
    The LangChain OCIModelDeployment integration lives in the `langchain-community` package. The following command will install `langchain-community` and the required dependencies.
    """
    logger.info("### Installation")
    
    # %pip install -qU langchain-community langchain-ollama oracle-ads
    
    """
    ## Instantiation
    
    You may instantiate the model with the generic `ChatOCIModelDeployment` or framework specific class like `ChatOCIModelDeploymentVLLM`.
    
    * Using `ChatOCIModelDeployment` when you need a generic entry point for deploying models. You can pass model parameters through `model_kwargs` during the instantiation of this class. This allows for flexibility and ease of configuration without needing to rely on framework-specific details.
    """
    logger.info("## Instantiation")
    
    
    chat = ChatOCIModelDeployment(
        endpoint="https://modeldeployment.<region>.oci.customer-oci.com/<ocid>/predict",
        streaming=True,
        max_retries=1,
        model_kwargs={
            "temperature": 0.2,
            "max_tokens": 512,
        },  # other model params...
        default_headers={
            "route": "/v1/chat/completions",
        },
    )
    
    """
    * Using framework specific class like `ChatOCIModelDeploymentVLLM`: This is suitable when you are working with a specific framework (e.g. `vLLM`) and need to pass model parameters directly through the constructor, streamlining the setup process.
    """
    
    
    chat = ChatOCIModelDeploymentVLLM(
        endpoint="https://modeldeployment.<region>.oci.customer-oci.com/<md_ocid>/predict",
    )
    
    """
    ## Invocation
    """
    logger.info("## Invocation")
    
    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]
    
    ai_msg = chat.invoke(messages)
    ai_msg
    
    logger.debug(ai_msg.content)
    
    """
    ## Chaining
    """
    logger.info("## Chaining")
    
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that translates {input_language} to {output_language}.",
            ),
            ("human", "{input}"),
        ]
    )
    
    chain = prompt | chat
    chain.invoke(
        {
            "input_language": "English",
            "output_language": "German",
            "input": "I love programming.",
        }
    )
    
    """
    ## Asynchronous calls
    """
    logger.info("## Asynchronous calls")
    
    
    system = "You are a helpful translator that translates {input_language} to {output_language}."
    human = "{text}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    
    chat = ChatOCIModelDeployment(
        endpoint="https://modeldeployment.us-ashburn-1.oci.customer-oci.com/<ocid>/predict"
    )
    chain = prompt | chat
    
    await chain.ainvoke(
        {
            "input_language": "English",
            "output_language": "Chinese",
            "text": "I love programming",
        }
    )
    
    """
    ## Streaming calls
    """
    logger.info("## Streaming calls")
    
    
    
    prompt = ChatPromptTemplate.from_messages(
        [("human", "List out the 5 states in the United State.")]
    )
    
    chat = ChatOCIModelDeployment(
        endpoint="https://modeldeployment.us-ashburn-1.oci.customer-oci.com/<ocid>/predict"
    )
    
    chain = prompt | chat
    
    for chunk in chain.stream({}):
        sys.stdout.write(chunk.content)
        sys.stdout.flush()
    
    """
    ## Structured output
    """
    logger.info("## Structured output")
    
    
    
    class Joke(BaseModel):
        """A setup to a joke and the punchline."""
    
        setup: str
        punchline: str
    
    
    chat = ChatOCIModelDeployment(
        endpoint="https://modeldeployment.us-ashburn-1.oci.customer-oci.com/<ocid>/predict",
    )
    structured_llm = chat.with_structured_output(Joke, method="json_mode")
    output = structured_llm.invoke(
        "Tell me a joke about cats, respond in JSON with `setup` and `punchline` keys"
    )
    
    output.dict()
    
    """
    ## API reference
    
    For comprehensive details on all features and configurations, please refer to the API reference documentation for each class:
    
    * [ChatOCIModelDeployment](https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.oci_data_science.ChatOCIModelDeployment.html)
    * [ChatOCIModelDeploymentVLLM](https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.oci_data_science.ChatOCIModelDeploymentVLLM.html)
    * [ChatOCIModelDeploymentTGI](https://python.langchain.com/api_reference/community/chat_models/langchain_community.chat_models.oci_data_science.ChatOCIModelDeploymentTGI.html)
    """
    logger.info("## API reference")
    
    logger.info("\n\n[DONE]", bright=True)

if __name__ == '__main__':
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())