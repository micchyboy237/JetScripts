async def main():
    from jet.transformers.formatters import format_json
    from jet.logger import CustomLogger
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.base.llms.types import ChatMessage
    from llama_index.core.tools import FunctionTool
    from llama_index.llms.oci_data_science import OCIDataScience
    import ads
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    """
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/oci_data_science.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    # Oracle Cloud Infrastructure Data Science 
    
    Oracle Cloud Infrastructure [(OCI) Data Science](https://www.oracle.com/artificial-intelligence/data-science) is a fully managed, serverless platform for data science teams to build, train, and manage machine learning models in Oracle Cloud Infrastructure.
    
    It offers [AI Quick Actions](https://docs.oracle.com/en-us/iaas/data-science/using/ai-quick-actions.htm), which can be used to deploy, evaluate, and fine-tune foundation LLM models in OCI Data Science. AI Quick Actions target users who want to quickly leverage the capabilities of AI. They aim to expand the reach of foundation models to a broader set of users by providing a streamlined, code-free, and efficient environment for working with foundation models. AI Quick Actions can be accessed from the Data Science Notebook.
    
    Detailed documentation on how to deploy LLM models in OCI Data Science using AI Quick Actions is available [here](https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/ai-quick-actions/model-deployment-tips.md) and [here](https://docs.oracle.com/en-us/iaas/data-science/using/ai-quick-actions-model-deploy.htm).
    
    This notebook explains how to use OCI's Data Science models with LlamaIndex.
    
    ## Setup
    
    If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
    """
    logger.info("# Oracle Cloud Infrastructure Data Science")
    
    # %pip install llama-index-llms-oci-data-science
    
    # !pip install llama-index
    
    """
    You will also need to install the [oracle-ads](https://accelerated-data-science.readthedocs.io/en/latest/index.html) SDK.
    """
    logger.info("You will also need to install the [oracle-ads](https://accelerated-data-science.readthedocs.io/en/latest/index.html) SDK.")
    
    # !pip install -U oracle-ads
    
    """
    ## Authentication
    The authentication methods supported for LlamaIndex are equivalent to those used with other OCI services and follow the standard SDK authentication methods, specifically API Key, session token, instance principal, and resource principal. More details can be found [here](https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/authentication.html). Make sure to have the required [policies](https://docs.oracle.com/en-us/iaas/data-science/using/model-dep-policies-auth.htm) to access the OCI Data Science Model Deployment endpoint. The [oracle-ads](https://accelerated-data-science.readthedocs.io/en/latest/index.html) helps to simplify the authentication within OCI Data Science.
    
    ## Basic Usage
    
    Using LLMs offered by OCI Data Science AI with LlamaIndex only requires you to initialize the `OCIDataScience` interface with your Data Science Model Deployment endpoint and model ID. By default the all deployed models in AI Quick Actions get `odsc-model` ID. However this ID cna be changed during the deployment.
    
    #### Call `complete` with a prompt
    """
    logger.info("## Authentication")
    
    
    ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")
    
    llm = OCIDataScience(
        model="odsc-llm",
        endpoint="https://<MD_OCID>/predict",
    )
    response = llm.complete("Tell me a joke")
    
    logger.debug(response)
    
    """
    ### Call `chat` with a list of messages
    """
    logger.info("### Call `chat` with a list of messages")
    
    
    ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")
    
    llm = OCIDataScience(
        model="odsc-llm",
        endpoint="https://<MD_OCID>/predict",
    )
    response = llm.chat(
        [
            ChatMessage(role="user", content="Tell me a joke"),
            ChatMessage(
                role="assistant", content="Why did the chicken cross the road?"
            ),
            ChatMessage(role="user", content="I don't know, why?"),
        ]
    )
    
    logger.debug(response)
    
    """
    ## Streaming
    
    ### Using `stream_complete` endpoint
    """
    logger.info("## Streaming")
    
    
    ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")
    
    llm = OCIDataScience(
        model="odsc-llm",
        endpoint="https://<MD_OCID>/predict",
    )
    
    for chunk in llm.stream_complete("Tell me a joke"):
        logger.debug(chunk.delta, end="")
    
    """
    ### Using `stream_chat` endpoint
    """
    logger.info("### Using `stream_chat` endpoint")
    
    
    ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")
    
    llm = OCIDataScience(
        model="odsc-llm",
        endpoint="https://<MD_OCID>/predict",
    )
    response = llm.stream_chat(
        [
            ChatMessage(role="user", content="Tell me a joke"),
            ChatMessage(
                role="assistant", content="Why did the chicken cross the road?"
            ),
            ChatMessage(role="user", content="I don't know, why?"),
        ]
    )
    
    for chunk in response:
        logger.debug(chunk.delta, end="")
    
    """
    ## Async
    
    ### Call `acomplete` with a prompt
    """
    logger.info("## Async")
    
    
    ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")
    
    llm = OCIDataScience(
        model="odsc-llm",
        endpoint="https://<MD_OCID>/predict",
    )
    response = llm.complete("Tell me a joke")
    logger.success(format_json(response))
    
    logger.debug(response)
    
    """
    ### Call `achat` with a list of messages
    """
    logger.info("### Call `achat` with a list of messages")
    
    
    ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")
    
    llm = OCIDataScience(
        model="odsc-llm",
        endpoint="https://<MD_OCID>/predict",
    )
    response = llm.chat(
            [
                ChatMessage(role="user", content="Tell me a joke"),
                ChatMessage(
                    role="assistant", content="Why did the chicken cross the road?"
                ),
                ChatMessage(role="user", content="I don't know, why?"),
            ]
        )
    logger.success(format_json(response))
    
    logger.debug(response)
    
    """
    ### Using `astream_complete` endpoint
    """
    logger.info("### Using `astream_complete` endpoint")
    
    
    ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")
    
    llm = OCIDataScience(
        model="odsc-llm",
        endpoint="https://<MD_OCID>/predict",
    )
    
    for chunk in llm.stream_complete("Tell me a joke"):
        logger.debug(chunk.delta, end="")
    
    """
    ### Using `astream_chat` endpoint
    """
    logger.info("### Using `astream_chat` endpoint")
    
    
    ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")
    
    llm = OCIDataScience(
        model="odsc-llm",
        endpoint="https://<MD_OCID>/predict",
    )
    response = await llm.stream_chat(
            [
                ChatMessage(role="user", content="Tell me a joke"),
                ChatMessage(
                    role="assistant", content="Why did the chicken cross the road?"
                ),
                ChatMessage(role="user", content="I don't know, why?"),
            ]
        )
    logger.success(format_json(response))
    
    async for chunk in response:
        logger.debug(chunk.delta, end="")
    
    """
    ## Configure Model
    """
    logger.info("## Configure Model")
    
    
    ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")
    
    llm = OCIDataScience(
        model="odsc-llm",
        endpoint="https://<MD_OCID>/predict",
        temperature=0.2,
        max_tokens=500,
        timeout=120,
        context_window=2500,
        additional_kwargs={
            "top_p": 0.75,
            "logprobs": True,
            "top_logprobs": 3,
        },
    )
    response = llm.chat(
        [
            ChatMessage(role="user", content="Tell me a joke"),
        ]
    )
    logger.debug(response)
    
    """
    ## Function Calling
    
    The [AI Quick Actions](https://docs.oracle.com/en-us/iaas/data-science/using/ai-quick-actions.htm) offers prebuilt service containers that make deploying and serving a large language model very easy. Either one of vLLM (a high-throughput and memory-efficient inference and serving engine for LLMs) or TGI (a high-performance text generation server for the popular open-source LLMs) is used in the service container to host the model, the end point created supports the OllamaFunctionCallingAdapter API protocol. This allows the model deployment to be used as a drop-in replacement for applications using OllamaFunctionCallingAdapter API. If the deployed model supports function calling, then integration with LlamaIndex tools, through the predict_and_call function on the llm allows to attach any tools and let the LLM decide which tools to call (if any).
    """
    logger.info("## Function Calling")
    
    
    ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")
    
    llm = OCIDataScience(
        model="odsc-llm",
        endpoint="https://<MD_OCID>/predict",
        temperature=0.2,
        max_tokens=500,
        timeout=120,
        context_window=2500,
        additional_kwargs={
            "top_p": 0.75,
            "logprobs": True,
            "top_logprobs": 3,
        },
    )
    
    
    def multiply(a: float, b: float) -> float:
        logger.debug(f"---> {a} * {b}")
        return a * b
    
    
    def add(a: float, b: float) -> float:
        logger.debug(f"---> {a} + {b}")
        return a + b
    
    
    def subtract(a: float, b: float) -> float:
        logger.debug(f"---> {a} - {b}")
        return a - b
    
    
    def divide(a: float, b: float) -> float:
        logger.debug(f"---> {a} / {b}")
        return a / b
    
    
    multiply_tool = FunctionTool.from_defaults(fn=multiply)
    add_tool = FunctionTool.from_defaults(fn=add)
    sub_tool = FunctionTool.from_defaults(fn=subtract)
    divide_tool = FunctionTool.from_defaults(fn=divide)
    
    response = llm.predict_and_call(
        [multiply_tool, add_tool, sub_tool, divide_tool],
        user_msg="Calculate the result of `8 + 2 - 6`.",
        verbose=True,
    )
    
    logger.debug(response)
    
    """
    ### Using `FunctionAgent`
    """
    logger.info("### Using `FunctionAgent`")
    
    
    ads.set_auth(auth="security_token", profile="<replace-with-your-profile>")
    
    llm = OCIDataScience(
        model="odsc-llm",
        endpoint="https://<MD_OCID>/predict",
        temperature=0.2,
        max_tokens=500,
        timeout=120,
        context_window=2500,
        additional_kwargs={
            "top_p": 0.75,
            "logprobs": True,
            "top_logprobs": 3,
        },
    )
    
    
    def multiply(a: float, b: float) -> float:
        logger.debug(f"---> {a} * {b}")
        return a * b
    
    
    def add(a: float, b: float) -> float:
        logger.debug(f"---> {a} + {b}")
        return a + b
    
    
    def subtract(a: float, b: float) -> float:
        logger.debug(f"---> {a} - {b}")
        return a - b
    
    
    def divide(a: float, b: float) -> float:
        logger.debug(f"---> {a} / {b}")
        return a / b
    
    
    multiply_tool = FunctionTool.from_defaults(fn=multiply)
    add_tool = FunctionTool.from_defaults(fn=add)
    sub_tool = FunctionTool.from_defaults(fn=subtract)
    divide_tool = FunctionTool.from_defaults(fn=divide)
    
    agent = FunctionAgent(
        tools=[multiply_tool, add_tool, sub_tool, divide_tool],
        llm=llm,
    )
    response = await agent.run(
            "Calculate the result of `8 + 2 - 6`. Use tools. Return the calculated result."
        )
    logger.success(format_json(response))
    
    logger.debug(response)
    
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