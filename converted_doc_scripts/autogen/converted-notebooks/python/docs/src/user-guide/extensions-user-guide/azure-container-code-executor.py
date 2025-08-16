import asyncio
from jet.transformers.formatters import format_json
from anyio import open_file
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors.azure import ACADynamicSessionsCodeExecutor
from azure.identity import DefaultAzureCredential
import os
import tempfile

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# ACA Dynamic Sessions Code Executor

This guide will explain the Azure Container Apps dynamic sessions in Azure Container Apps and show you how to use the Azure Container Code Executor class.

The [Azure Container Apps dynamic sessions](https://learn.microsoft.com/en-us/azure/container-apps/sessions) is a component in the Azure Container Apps service. The environment is hosted on remote Azure instances and will not execute any code locally. The interpreter is capable of executing python code in a jupyter environment with a pre-installed base of commonly used packages. [Custom environments](https://learn.microsoft.com/en-us/azure/container-apps/sessions-custom-container) can be created by users for their applications. Files can additionally be [uploaded to, or downloaded from](https://learn.microsoft.com/en-us/azure/container-apps/sessions-code-interpreter#upload-a-file-to-a-session) each session.

The code interpreter can run multiple sessions of code, each of which are delineated by a session identifier string.

## Create a Container Apps Session Pool

In your Azure portal, create a new `Container App Session Pool` resource with the pool type set to `Python code interpreter` and note the `Pool management endpoint`. The format for the endpoint should be something like `https://{region}.dynamicsessions.io/subscriptions/{subscription_id}/resourceGroups/{resource_group_name}/sessionPools/{session_pool_name}`.

Alternatively, you can use the [Azure CLI to create a session pool.](https://learn.microsoft.com/en-us/azure/container-apps/sessions-code-interpreter#create-a-session-pool-with-azure-cli)

## ACADynamicSessionsCodeExecutor

The {py:class}`~autogen_ext.code_executors.azure.ACADynamicSessionsCodeExecutor` class is a python code executor that creates and executes arbitrary python code on a default Serverless code interpreter session. Its interface is as follows

### Initialization

First, you will need to find or create a credentialing object that implements the {py:class}`~autogen_ext.code_executors.azure.TokenProvider` interface. This is any object that implements the following function
```python
def get_token(
    self, *scopes: str, claims: Optional[str] = None, tenant_id: Optional[str] = None, **kwargs: Any
) -> azure.core.credentials.AccessToken
```
An example of such an object is the [azure.identity.DefaultAzureCredential](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python) class.

Lets start by installing that
"""
logger.info("# ACA Dynamic Sessions Code Executor")



"""
Next, lets import all the necessary modules and classes for our code
"""
logger.info("Next, lets import all the necessary modules and classes for our code")



"""
Now, we create our Azure code executor and run some test code along with verification that it ran correctly. We'll create the executor with a temporary working directory to ensure a clean environment as we show how to use each feature
"""
logger.info("Now, we create our Azure code executor and run some test code along with verification that it ran correctly. We'll create the executor with a temporary working directory to ensure a clean environment as we show how to use each feature")

cancellation_token = CancellationToken()
POOL_MANAGEMENT_ENDPOINT = "..."

with tempfile.TemporaryDirectory() as temp_dir:
    executor = ACADynamicSessionsCodeExecutor(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT, credential=DefaultAzureCredential(), work_dir=temp_dir
    )

    code_blocks = [CodeBlock(code="import sys; logger.debug('hello world!')", language="python")]
    async def run_async_code_0a5a8bfd():
        async def run_async_code_1382bdfc():
            code_result = await executor.execute_code_blocks(code_blocks, cancellation_token)
            return code_result
        code_result = asyncio.run(run_async_code_1382bdfc())
        logger.success(format_json(code_result))
        return code_result
    code_result = asyncio.run(run_async_code_0a5a8bfd())
    logger.success(format_json(code_result))
    assert code_result.exit_code == 0 and "hello world!" in code_result.output

"""
Next, lets try uploading some files and verifying their integrity. All files uploaded to the Serverless code interpreter is uploaded into the `/mnt/data` directory. All downloadable files must also be placed in the directory. By default, the current working directory for the code executor is set to `/mnt/data`.
"""
logger.info("Next, lets try uploading some files and verifying their integrity. All files uploaded to the Serverless code interpreter is uploaded into the `/mnt/data` directory. All downloadable files must also be placed in the directory. By default, the current working directory for the code executor is set to `/mnt/data`.")

with tempfile.TemporaryDirectory() as temp_dir:
    test_file_1 = "test_upload_1.txt"
    test_file_1_contents = "test1 contents"
    test_file_2 = "test_upload_2.txt"
    test_file_2_contents = "test2 contents"

    async def async_func_6():
        async with await open_file(os.path.join(temp_dir, test_file_1), "w") as f:  # type: ignore[syntax]
            await f.write(test_file_1_contents)
        return result

    result = asyncio.run(async_func_6())
    logger.success(format_json(result))
    async def async_func_8():
        async with await open_file(os.path.join(temp_dir, test_file_2), "w") as f:  # type: ignore[syntax]
            await f.write(test_file_2_contents)
            
        return result

    result = asyncio.run(async_func_8())
    logger.success(format_json(result))
    assert os.path.isfile(os.path.join(temp_dir, test_file_1))
    assert os.path.isfile(os.path.join(temp_dir, test_file_2))

    executor = ACADynamicSessionsCodeExecutor(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT, credential=DefaultAzureCredential(), work_dir=temp_dir
    )
    async def run_async_code_a1bf142d():
        await executor.upload_files([test_file_1, test_file_2], cancellation_token)
        return 
     = asyncio.run(run_async_code_a1bf142d())
    logger.success(format_json())

    async def run_async_code_2dc6a69f():
        async def run_async_code_9e45292e():
            file_list = await executor.get_file_list(cancellation_token)
            return file_list
        file_list = asyncio.run(run_async_code_9e45292e())
        logger.success(format_json(file_list))
        return file_list
    file_list = asyncio.run(run_async_code_2dc6a69f())
    logger.success(format_json(file_list))
    assert test_file_1 in file_list
    assert test_file_2 in file_list

    code_blocks = [
        CodeBlock(
            code=f"""
with open("{test_file_1}") as f:
  logger.debug(f.read())
with open("{test_file_2}") as f:
  logger.debug(f.read())
""",
            language="python",
        )
    ]
    async def run_async_code_0a5a8bfd():
        async def run_async_code_1382bdfc():
            code_result = await executor.execute_code_blocks(code_blocks, cancellation_token)
            return code_result
        code_result = asyncio.run(run_async_code_1382bdfc())
        logger.success(format_json(code_result))
        return code_result
    code_result = asyncio.run(run_async_code_0a5a8bfd())
    logger.success(format_json(code_result))
    assert code_result.exit_code == 0
    assert test_file_1_contents in code_result.output
    assert test_file_2_contents in code_result.output

"""
Downloading files works in a similar way.
"""
logger.info("Downloading files works in a similar way.")

with tempfile.TemporaryDirectory() as temp_dir:
    test_file_1 = "test_upload_1.txt"
    test_file_1_contents = "test1 contents"
    test_file_2 = "test_upload_2.txt"
    test_file_2_contents = "test2 contents"

    assert not os.path.isfile(os.path.join(temp_dir, test_file_1))
    assert not os.path.isfile(os.path.join(temp_dir, test_file_2))

    executor = ACADynamicSessionsCodeExecutor(
        pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT, credential=DefaultAzureCredential(), work_dir=temp_dir
    )

    code_blocks = [
        CodeBlock(
            code=f"""
with open("{test_file_1}", "w") as f:
  f.write("{test_file_1_contents}")
with open("{test_file_2}", "w") as f:
  f.write("{test_file_2_contents}")
""",
            language="python",
        ),
    ]
    async def run_async_code_0a5a8bfd():
        async def run_async_code_1382bdfc():
            code_result = await executor.execute_code_blocks(code_blocks, cancellation_token)
            return code_result
        code_result = asyncio.run(run_async_code_1382bdfc())
        logger.success(format_json(code_result))
        return code_result
    code_result = asyncio.run(run_async_code_0a5a8bfd())
    logger.success(format_json(code_result))
    assert code_result.exit_code == 0

    async def run_async_code_2dc6a69f():
        async def run_async_code_9e45292e():
            file_list = await executor.get_file_list(cancellation_token)
            return file_list
        file_list = asyncio.run(run_async_code_9e45292e())
        logger.success(format_json(file_list))
        return file_list
    file_list = asyncio.run(run_async_code_2dc6a69f())
    logger.success(format_json(file_list))
    assert test_file_1 in file_list
    assert test_file_2 in file_list

    async def run_async_code_b5f2163a():
        await executor.download_files([test_file_1, test_file_2], cancellation_token)
        return 
     = asyncio.run(run_async_code_b5f2163a())
    logger.success(format_json())

    assert os.path.isfile(os.path.join(temp_dir, test_file_1))
    async def async_func_34():
        async with await open_file(os.path.join(temp_dir, test_file_1), "r") as f:  # type: ignore[syntax]
            async def run_async_code_8f4f39b7():
                content = await f.read()
                return content
            content = asyncio.run(run_async_code_8f4f39b7())
            logger.success(format_json(content))
            assert test_file_1_contents in content
        return result

    result = asyncio.run(async_func_34())
    logger.success(format_json(result))
    assert os.path.isfile(os.path.join(temp_dir, test_file_2))
    async def async_func_38():
        async with await open_file(os.path.join(temp_dir, test_file_2), "r") as f:  # type: ignore[syntax]
            async def run_async_code_8f4f39b7():
                content = await f.read()
                return content
            content = asyncio.run(run_async_code_8f4f39b7())
            logger.success(format_json(content))
            assert test_file_2_contents in content
        return result

    result = asyncio.run(async_func_38())
    logger.success(format_json(result))

"""
### New Sessions

Every instance of the {py:class}`~autogen_ext.code_executors.azure.ACADynamicSessionsCodeExecutor` class will have a unique session ID. Every call to a particular code executor will be executed on the same session until the {py:meth}`~autogen_ext.code_executors.azure.ACADynamicSessionsCodeExecutor.restart` function is called on it. Previous sessions cannot be reused.

Here we'll run some code on the code session, restart it, then verify that a new session has been opened.
"""
logger.info("### New Sessions")

executor = ACADynamicSessionsCodeExecutor(
    pool_management_endpoint=POOL_MANAGEMENT_ENDPOINT, credential=DefaultAzureCredential()
)

code_blocks = [CodeBlock(code="x = 'abcdefg'", language="python")]
async def run_async_code_feb2652e():
    async def run_async_code_0a5a8bfd():
        code_result = await executor.execute_code_blocks(code_blocks, cancellation_token)
        return code_result
    code_result = asyncio.run(run_async_code_0a5a8bfd())
    logger.success(format_json(code_result))
    return code_result
code_result = asyncio.run(run_async_code_feb2652e())
logger.success(format_json(code_result))
assert code_result.exit_code == 0

code_blocks = [CodeBlock(code="logger.debug(x)", language="python")]
async def run_async_code_feb2652e():
    async def run_async_code_0a5a8bfd():
        code_result = await executor.execute_code_blocks(code_blocks, cancellation_token)
        return code_result
    code_result = asyncio.run(run_async_code_0a5a8bfd())
    logger.success(format_json(code_result))
    return code_result
code_result = asyncio.run(run_async_code_feb2652e())
logger.success(format_json(code_result))
assert code_result.exit_code == 0 and "abcdefg" in code_result.output

async def run_async_code_fa0bf6d3():
    await executor.restart()
    return 
 = asyncio.run(run_async_code_fa0bf6d3())
logger.success(format_json())
code_blocks = [CodeBlock(code="logger.debug(x)", language="python")]
async def run_async_code_feb2652e():
    async def run_async_code_0a5a8bfd():
        code_result = await executor.execute_code_blocks(code_blocks, cancellation_token)
        return code_result
    code_result = asyncio.run(run_async_code_0a5a8bfd())
    logger.success(format_json(code_result))
    return code_result
code_result = asyncio.run(run_async_code_feb2652e())
logger.success(format_json(code_result))
assert code_result.exit_code != 0 and "NameError" in code_result.output

"""
### Available Packages

Each code execution instance is pre-installed with most of the commonly used packages. However, the list of available packages and versions are not available outside of the execution environment. The packages list on the environment can be retrieved by calling the {py:meth}`~autogen_ext.code_executors.azure.ACADynamicSessionsCodeExecutor.get_available_packages` function on the code executor.
"""
logger.info("### Available Packages")

logger.debug(executor.get_available_packages(cancellation_token))

logger.info("\n\n[DONE]", bright=True)