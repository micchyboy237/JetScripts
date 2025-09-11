from jet.logger import logger
from langchain_community.llms import OCIModelDeploymentLLM
from langchain_community.llms import OCIModelDeploymentTGI
from langchain_community.llms import OCIModelDeploymentVLLM
import ads
import os
import shutil

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
    # OCI Data Science Model Deployment Endpoint
    
    [OCI Data Science](https://docs.oracle.com/en-us/iaas/data-science/using/home.htm) is a fully managed and serverless platform for data science teams to build, train, and manage machine learning models in the Oracle Cloud Infrastructure.
    
    > For the latest updates, examples and experimental features, please see [ADS LangChain Integration](https://accelerated-data-science.readthedocs.io/en/latest/user_guide/large_language_model/langchain_models.html).
    
    This notebooks goes over how to use an LLM hosted on a [OCI Data Science Model Deployment](https://docs.oracle.com/en-us/iaas/data-science/using/model-dep-about.htm).
    
    For authentication, the [oracle-ads](https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/authentication.html) library is used to automatically load credentials required for invoking the endpoint.
    """
    logger.info("# OCI Data Science Model Deployment Endpoint")
    
    # !pip3 install oracle-ads
    
    """
    ## Prerequisite
    
    ### Deploy model
    You can easily deploy, fine-tune, and evaluate foundation models using the [AI Quick Actions](https://docs.oracle.com/en-us/iaas/data-science/using/ai-quick-actions.htm) on OCI Data Science Model deployment. For additional deployment examples, please visit the [Oracle GitHub samples repository](https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/ai-quick-actions/llama3-with-smc.md). 
    
    ### Policies
    Make sure to have the required [policies](https://docs.oracle.com/en-us/iaas/data-science/using/model-dep-policies-auth.htm#model_dep_policies_auth__predict-endpoint) to access the OCI Data Science Model Deployment endpoint.
    
    ## Set up
    
    After having deployed model, you have to set up following required parameters of the call:
    
    - **`endpoint`**: The model HTTP endpoint from the deployed model, e.g. `https://modeldeployment.<region>.oci.customer-oci.com/<md_ocid>/predict`. 
    
    
    ### Authentication
    
    You can set authentication through either ads or environment variables. When you are working in OCI Data Science Notebook Session, you can leverage resource principal to access other OCI resources. Check out [here](https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/authentication.html) to see more options. 
    
    ## Examples
    """
    logger.info("## Prerequisite")
    
    
    ads.set_auth("resource_principal")
    
    llm = OCIModelDeploymentLLM(
        endpoint="https://modeldeployment.<region>.oci.customer-oci.com/<md_ocid>/predict",
        model="odsc-llm",
    )
    
    llm.invoke("Who is the first president of United States?")
    
    
    ads.set_auth("resource_principal")
    
    llm = OCIModelDeploymentVLLM(
        endpoint="https://modeldeployment.<region>.oci.customer-oci.com/<md_ocid>/predict",
    )
    
    llm.invoke("Who is the first president of United States?")
    
    
    
    os.environ["OCI_IAM_TYPE"] = "api_key"
    os.environ["OCI_CONFIG_PROFILE"] = "default"
    os.environ["OCI_CONFIG_LOCATION"] = "~/.oci"
    
    os.environ["OCI_LLM_ENDPOINT"] = (
        "https://modeldeployment.<region>.oci.customer-oci.com/<md_ocid>/predict"
    )
    
    llm = OCIModelDeploymentTGI()
    
    llm.invoke("Who is the first president of United States?")
    
    """
    ### Asynchronous calls
    """
    logger.info("### Asynchronous calls")
    
    await llm.ainvoke("Tell me a joke.")
    
    """
    ### Streaming calls
    """
    logger.info("### Streaming calls")
    
    for chunk in llm.stream("Tell me a joke."):
        logger.debug(chunk, end="", flush=True)
    
    """
    ## API reference
    
    For comprehensive details on all features and configurations, please refer to the API reference documentation for each class:
    
    * [OCIModelDeploymentLLM](https://python.langchain.com/api_reference/community/llms/langchain_community.llms.oci_data_science_model_deployment_endpoint.OCIModelDeploymentLLM.html)
    * [OCIModelDeploymentVLLM](https://python.langchain.com/api_reference/community/llms/langchain_community.llms.oci_data_science_model_deployment_endpoint.OCIModelDeploymentVLLM.html)
    * [OCIModelDeploymentTGI](https://python.langchain.com/api_reference/community/llms/langchain_community.llms.oci_data_science_model_deployment_endpoint.OCIModelDeploymentTGI.html)
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